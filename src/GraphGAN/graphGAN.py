import torch
import torch.nn as nn
import os
import sys
import tqdm
import pickle
import numpy as np
import collections
from discriminator import Discriminator
from generator import Generator
import config
from src import utils
from src import rcmd_util
from src.evaluation import link_prediction as lp
from src.evaluation import node_classification as nc
from BFS_trees import BFS_trees


class graphGAN():
    def __init__(self):
        utils.make_config_dirs(config)

        if config.app == "link_prediction":
            self.graph = utils.read_edges(train_filename=config.train_filename, test_filename=config.test_filename)
            self.n_node = max(list(self.graph.keys())) + 1
        elif config.app == "node_classification":
            self.graph = utils.read_edges(train_filename=config.train_filename)
            self.n_node = max(list(self.graph.keys())) + 1
            self.n_classes ,self.labels_matrix = utils.read_labels(filename=config.labels_filename, n_node=self.n_node)
        elif config.app == "recommendation":
            self.graph, self.rcmd = rcmd_util.read_edges(train_filename=config.rcmd_train_filename, 
                                                         test_filename=config.rcmd_test_filename)
            self.n_node = max(list(self.graph.keys())) + 1                                       
        else:
            raise Exception("Unknown task: {}".format(config.app))

        if config.app == "recommendation":
            self.root_nodes = sorted(list(self.graph.keys()))[:self.rcmd.user_max]
        else:
            self.root_nodes = sorted(list(self.graph.keys()))

        node_embed_init_d = utils.read_embeddings(filename=config.pretrain_emb_filename_d,
                                                       n_node=self.n_node,
                                                       n_embed=config.n_emb)
        node_embed_init_g = utils.read_embeddings(filename=config.pretrain_emb_filename_g,
                                                       n_node=self.n_node,
                                                       n_embed=config.n_emb)
        self.discriminator = Discriminator(n_node=self.n_node, node_emd_init=node_embed_init_d)
        self.generator = Generator(n_node=self.n_node, node_emd_init=node_embed_init_g)

        if config.app == "recommendation":
            self.BFS_trees = BFS_trees(self.root_nodes, self.graph, batch_num=config.cache_batch, 
                                       app=config.app, rcmd=self.rcmd)
        else:
            self.BFS_trees = BFS_trees(self.root_nodes, self.graph, batch_num=config.cache_batch)
        
        

    def prepare_data_for_d(self):
        print("prepare_data_for_d")
        center_nodes = []
        neighbor_nodes = []
        labels = []
        none_cnt = 0
        for i in tqdm.tqdm(self.root_nodes):
            if np.random.rand() < config.update_ratio:
                pos = self.graph[i]
                neg, _ = self.sample(i, self.BFS_trees.get_tree(i), len(pos), for_d=True)
                if neg is None:
                    none_cnt += 1
                if len(pos) != 0 and neg is not None:
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))

                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(neg)
                    labels.extend([0] * len(neg))
        return center_nodes, neighbor_nodes, labels

    def prepare_data_for_g(self):
        print("prepare_data_for_g")
        paths = []
        for i in tqdm.tqdm(self.root_nodes):
            if np.random.rand() < config.update_ratio:
                sample, paths_from_i = self.sample(i, self.BFS_trees.get_tree(i), config.n_sample_gen, for_d=False)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])

        reward = self.discriminator.reward(node_1, node_2)
        return node_1, node_2, reward
    

    def sample(self, root, tree, sample_num, for_d):

        all_score = self.generator.all_score().numpy()
        samples = []
        paths = []
        n = 0

        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:
                    return None, None
                if for_d:
                    if node_neighbor == [root]:
                        return None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                relevance_probability = all_score[current_node, node_neighbor]
                relevance_probability = utils.softmax(relevance_probability)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]
                paths[n].append(next_node)
                if next_node == previous_node:
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths

    @staticmethod
    def get_node_pairs_from_path(path):

        path = path[:-1]
        pairs = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - config.window_size, 0), min(i + config.window_size + 1, len(path))):
                if i == j:
                    continue
                node = path[j]
                pairs.append([center_node, node])
        return pairs
    

    def write_embeddings_to_file(self):

        modes = [self.generator, self.discriminator]
        for i in range(2):
            embedding_matrix = modes[i].embedding_matrix.detach().numpy()
            index = np.array(range(self.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for emb in embedding_list]
            with open(config.emb_filenames[i], "w+") as f:
                lines = [str(self.n_node) + "\t" + str(config.n_emb) + "\n"] + embedding_str
                f.writelines(lines)
    