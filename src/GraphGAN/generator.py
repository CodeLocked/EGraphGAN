import torch
import torch.nn as nn
import config
from sklearn.neighbors import KernelDensity
import numpy as np
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, n_node, node_emd_init):
        super(Generator, self).__init__()
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        self.embedding_matrix_minimax = nn.Parameter(torch.tensor(node_emd_init))
        self.bias_minimax = nn.Parameter(torch.zeros([self.n_node]))

        self.embedding_matrix_heuristic = nn.Parameter(torch.tensor(node_emd_init))
        self.bias_heuristic = nn.Parameter(torch.zeros([self.n_node]))

        self.embedding_matrix_least_squares = nn.Parameter(torch.tensor(node_emd_init))
        self.bias_least_squares = nn.Parameter(torch.zeros([self.n_node]))

        self.embedding_matrix = nn.Parameter(torch.tensor(node_emd_init))
        self.bias = nn.Parameter(torch.zeros([self.n_node]))
        
        self.node_embedding = None
        self.node_neighbor_embedding = None

    def forward(self, nu_d, theta_g):
        mutated_x1 = self.minimax_mutation(self.embedding_matrix_minimax, nu_d, theta_g)
        mutated_x2 = self.heuristic_mutation(self.embedding_matrix_heuristic)
        mutated_x3 = self.least_squares_mutation(self.embedding_matrix_least_squares)
        concatenated_matrices = torch.stack([mutated_x1, mutated_x2, mutated_x3], dim=-1)
        matrix_embedding = self.generator_model(concatenated_matrices, nu_d, theta_g)
        return matrix_embedding

    def minimax_mutation(self, x, nu_d, theta_g, epsilon=1e-8):
        p_data = self.calculate_true_data_distribution(x)
        G_distribution = self.calculate_generator_distribution(nu_d, theta_g)
        jensen_shannon_divergence = self.calculate_jensen_shannon_divergence(p_data, G_distribution, epsilon)
        min_val = torch.min(x)
        max_val = torch.max(x)
        mutated_x1 = x * (1 - jensen_shannon_divergence) + (min_val + max_val) * jensen_shannon_divergence
        return mutated_x1

    def calculate_true_data_distribution(self, x):
        discrete_distribution = self.calculate_discrete_distribution(x)
        continuous_distribution = self.kernel_density_estimation(x)
        p_data = 0.7 * discrete_distribution + 0.3 * continuous_distribution
        return p_data

    def kernel_density_estimation(self):
        x_np = self.numpy() if isinstance(self, torch.Tensor) else self
        if len(x_np.shape) == 1:
            x_np = x_np.reshape(-1, 1)
        kde = KernelDensity(bandwidth=0.6, kernel='gaussian')
        kde.fit(x_np)
        x_range = np.linspace(np.min(x_np), np.max(x_np), 1000).reshape(-1, 1)
        log_density = kde.score_samples(x_range)
        estimated_density = np.exp(log_density)
        return torch.from_numpy(estimated_density) if isinstance(self, torch.Tensor) else estimated_density

    def calculate_generator_distribution(self, nu_d, theta_g):
        generated_samples = self.generator_model(nu_d, theta_g)
        discrete_distribution = self.calculate_discrete_distribution(generated_samples)
        continuous_distribution = self.kernel_density_estimation(generated_samples)
        g_distribution = 0.7 * discrete_distribution + 0.3 * continuous_distribution
        return g_distribution

    def calculate_jensen_shannon_divergence(self, p, q, epsilon=1e-8):
        m = 0.5 * (p + q)
        kl_divergence_p = F.kl_div(p.log(), m, reduction='batchmean')
        kl_divergence_q = F.kl_div(q.log(), m, reduction='batchmean')
        js_divergence = 0.5 * (kl_divergence_p + kl_divergence_q + epsilon)
        return js_divergence

    def calculate_discrete_distribution(self, x):
        unique_values, counts = torch.unique(x, return_counts=True)
        discrete_distribution = counts / torch.sum(counts)
        return discrete_distribution

    def heuristic_mutation(self, x):
        x = torch.abs(x)
        non_linear_values = x ** 2
        sin_values = torch.sin(x)
        min_val = torch.min(x)
        max_val = torch.max(x)
        scaling_factor = (max_val - min_val) / (min_val + max_val + 1e-8)
        mutated_x2 = 0.4 * non_linear_values + 0.3 * sin_values + 0.3 * scaling_factor * x
        return mutated_x2

    def least_squares_mutation(self, x):
        x = torch.abs(x)
        non_linear_values = torch.sqrt(x)
        power_values = x**0.8
        min_val = torch.min(x)
        max_val = torch.max(x)
        scaling_factor = (max_val - min_val) / (min_val + max_val + 1e-8)
        mutated_x3 = 0.4 * non_linear_values + 0.3 * power_values + 0.3 * scaling_factor * (x - 1.0)
        return mutated_x3

    def all_score(self):
        return torch.matmul(self.embedding_matrix, torch.transpose(self.embedding_matrix, 0, 1)).detach()
    
    def score(self, node_id, node_neighbor_id):
        self.node_embedding = self.embedding_matrix[node_id, :]
        self.node_neighbor_embedding = self.embedding_matrix[node_neighbor_id, :]
        return torch.sum(input=self.node_embedding * self.node_neighbor_embedding, dim=1) + self.bias[node_neighbor_id]
    
    def loss(self, prob, reward):
        l2_loss = lambda x: torch.sum(x * x) / 2 * config.lambda_gen
        prob = torch.clamp(input=prob, min=1e-5, max=1)
        regularization = l2_loss(self.node_embedding) + l2_loss(self.node_neighbor_embedding)
        _loss = -torch.mean(torch.log(prob) * reward) + regularization

        return _loss

if __name__ == "__main__":
    import sys, os
    sys.path.append("../..")
    os.chdir(sys.path[0]) 
    from src import utils
    n_node, graph = utils.read_edges(train_filename=config.train_filename, test_filename=config.test_filename)
    node_embed_init_g = utils.read_embeddings(filename=config.pretrain_emb_filename_g,
                                              n_node=n_node,
                                              n_embed=config.n_emb)
    generator = Generator(n_node=n_node, node_emd_init=node_embed_init_g)
    for p in generator.parameters():
        print(p.name)
        print(p)
    