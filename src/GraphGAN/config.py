modes = ["gen", "dis"]

# training settings
batch_size_gen = 64  # batch size for the generator
batch_size_dis = 64  # batch size for the discriminator
lambda_gen = 1e-5  # l2 loss regulation weight for the generator
lambda_dis = 1e-5  # l2 loss regulation weight for the discriminator
n_sample_gen = 20  # number of samples for the generator
lr_gen = 1e-6  # learning rate for the generator
lr_dis = 1e-6  # learning rate for the discriminator
n_epochs = 500  # number of outer loops
n_epochs_gen = 550  # number of inner loops for the generator
n_epochs_dis = 550  # number of inner loops for the discriminator
gen_interval = n_epochs_gen  # sample new nodes for the generator for every gen_interval iterations
dis_interval = n_epochs_dis  # sample new nodes for the discriminator for every dis_interval iterations
update_ratio = 1    # updating ratio when choose the trees

rcmd_K = (2,10,20,50,100)

# model saving
load_model = False  # whether loading existing model for initialization
save_steps = 10

# other hyper-parameters
n_emb = 50
multi_processing = False  # whether using multi-processing to construct BFS-trees
window_size = 2

# application and dataset settings
app = ["link_prediction", "node_classification", "recommendation"][2]
dataset = ["CA-GrQc", "BlogCatalog", "MovieLens-1M"][2]
cache_batch = [2, 10, 5][2]

# path settings
data_root = "../../data/" + app + "/" + dataset + "/"

#for link prediction
train_filename = data_root + "train.txt"
test_filename = data_root + "test.txt"
test_neg_filename = data_root + "test_neg.txt"

#for node classification
neg_filename = data_root + "neg.txt"
labels_filename = data_root + "labels.txt"

#for recommendation
rcmd_train_filename = data_root + "train.csv"
rcmd_test_filename = data_root + "test.csv"
rcmd_test_neg_filename = data_root + "test_neg.csv"

#pretrain_embedding
pretrain_emb_filename_d = "/home/pilab/wzy/test/pre_train/" + app + "/" + dataset + "_pre_train.emb"
pretrain_emb_filename_g = "/home/pilab/wzy/test/pre_train/" + app + "/" + dataset + "_pre_train.emb"

#embedding
emb_filenames = ["/home/pilab/wzy/test/results/" + app + "/" + dataset + "_gen_.emb",
                 "/home/pilab/wzy/test/results/" + app + "/" + dataset + "_dis_.emb"]

result_filename = "/home/pilab/wzy/test/results/" + app + "/" + dataset + ".txt"

cache_filename = "/home/pilab/wzy/test/cache/" + app + "/" + dataset + ".pkl"

train_detail_filename = ["/home/pilab/wzy/test/results/" + app + "/" + "gen_detail/" + dataset + ".txt",
                         "/home/pilab/wzy/test/results/" + app + "/" + "dis_detail/" + dataset + ".txt"]

#model_log = "../../log/"
dis_state_dict_filename = "/home/pilab/wzy/test/model_params/" + dataset + "_dis.pt"
gen_state_dict_filename = "/home/pilab/wzy/test/model_params/" + dataset + "_gen.pt"
