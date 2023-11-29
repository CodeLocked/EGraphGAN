import os
import sys
import tqdm
import pickle
import collections
import config

class BFS_trees:
    def __init__(self, root_nodes, graph, batch_num=1, app=None, rcmd=None):
        self.app = app
        #self.root_nodes = root_nodes
        self.graph = graph
        self.batch_num = batch_num
        self.index = -1
        self.step = int(len(root_nodes) / self.batch_num)

        #create roots list
        self.roots_lst = []
        for i in range(self.batch_num):
            self.roots_lst.append(root_nodes[i*self.step : (i+1)*self.step])
    
        if self.batch_num * self.step < len(root_nodes):
            self.roots_lst.append(root_nodes[self.batch_num*self.step : len(root_nodes)])
        
        #for distinguish between users and movies
        if self.app == "recommendation":
            self.rcmd = rcmd
        
        self.BFS_trees = None
        
    
    def read_from_file(self, filename, present_nodes):
        BFS_trees = None
        if os.path.isfile(filename):
            print("reading BFS-trees from cache ... " + str(self.index))
            pickle_file = open(filename, 'rb')
            BFS_trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructiong BFS-trees " + str(self.index))

            if self.app == "recommendation":
                BFS_trees = self.construct_trees_rcmd(present_nodes)
            else:
                BFS_trees = self.construct_trees(present_nodes)

            pickle_file = open(filename, 'wb')
            pickle.dump(BFS_trees, pickle_file)
            pickle_file.close()
        return BFS_trees

    def out_of_bound(self, root, roots):
        return root < roots[0] or root > roots[-1]

    def get_tree(self, root):
        if self.index < 0 or self.out_of_bound(root, self.roots_lst[self.index]):
            for i, roots in enumerate(self.roots_lst):
                if self.out_of_bound(root, roots) == False:
                    self.index = i
                    break
            
            filename = config.cache_filename + '_' + str(self.index)
            present_nodes = self.roots_lst[self.index]

            self.BFS_trees = self.read_from_file(filename, present_nodes)

        return self.BFS_trees[root]
    

    def construct_trees(self, nodes):
        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in self.graph[cur_node]:
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees
    

    def construct_trees_rcmd(self, nodes):
        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in self.graph[cur_node]:
                    if sub_node in used_nodes:
                        continue
                    if self.rcmd.is_user(sub_node):
                        for m_node in self.graph[sub_node]:
                            if m_node not in used_nodes:
                                trees[root][cur_node].append(m_node)
                                trees[root][m_node] = [cur_node]
                                queue.append(m_node)
                                used_nodes.add(m_node)
                    else:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                    used_nodes.add(sub_node)
        return trees