import PhysicsProcess as pp
from FeynGraph import FeynGraph
import numpy as np
import torch 
import pandas as pd
import os 
import glob
import random as rd


class GraphSet:
    def __init__(self, proc_list=pp.PROC_LIST):
        self.proc_list = proc_list
        self.adj = []
        self.nodes_feat = []
        self.amp = []
        self.size = 0

    def creator(
            self, size:int=10000, seed = None
        ):
        self.size = len(self.proc_list)*4*size
        for t in range(len(self.proc_list)):
            proc = getattr(pp, self.proc_list[t])
            for i in range(size):
                adj, node_feat, amp = proc('electron', 1)
                self.adj.append(adj)
                self.nodes_feat.append(node_feat)
                self.amp.append(amp)
                adj, node_feat, amp = proc('electron', -1)
                self.adj.append(adj)
                self.nodes_feat.append(node_feat)
                self.amp.append(amp)
                adj, node_feat, amp = proc('muon', 1)
                self.adj.append(adj)
                self.nodes_feat.append(node_feat)
                self.amp.append(amp)
                adj, node_feat, amp = proc('muon', 1)
                self.adj.append(adj)
                self.nodes_feat.append(node_feat)
                self.amp.append(amp)

    def saver(self, dir_root):
        for p in range(len(self.proc_list)):
            dic = {
                'num_nodes': [], 
                'nodes_feat': [],
                'adj': [],
                'amp': self.amp
            }
            for i in range(self.size):
                dic['num_nodes'].append(self.adj[i].shape[0])
                dic['nodes_feat'].append(self.nodes_feat[i].flatten())
                dic['adj'].append(self.adj[i].flatten())
            pd.DataFrame(dic).to_csv(dir_root+'/test.csv', index=False)

    def spliter(self, ratio=(8,1,1),seed=100):
        train_gs, test_gs, validate_gs = GraphSet(self.proc_list),GraphSet(self.proc_list),GraphSet(self.proc_list)
        torch.manual_seed(seed)
        rand_id = torch.randperm(self.size)
        p1 = int(ratio[0]/(ratio[0]+ratio[1]+ratio[2])*self.size)
        p2 = int((ratio[0]+ratio[1])/(ratio[0]+ratio[1]+ratio[2])*self.size)
        for i in range(len(self.proc_list)):
            #train
            train_gs.feat_nodes[i] = self.feat_nodes[i][rand_id[:p1]]
            train_gs.feat_edges[i] = self.feat_edges[i][rand_id[:p1]]
            train_gs.amp[i] = self.amp[i][rand_id[:p1]]
            #test
            test_gs.feat_nodes[i] = self.feat_nodes[i][rand_id[p1:p2]]
            test_gs.feat_edges[i] = self.feat_edges[i][rand_id[p1:p2]]
            test_gs.amp[i] = self.amp[i][rand_id[p1:p2]]
            #validate
            validate_gs.feat_nodes[i] = self.feat_nodes[i][rand_id[p2:]]
            validate_gs.feat_edges[i] = self.feat_edges[i][rand_id[p2:]]
            validate_gs.amp[i] = self.amp[i][rand_id[p2:]]
        train_gs.size = len(train_gs.feat_nodes[0])
        test_gs.size = len(test_gs.feat_nodes[0])
        validate_gs.size = len(validate_gs.feat_nodes[0])
        return train_gs,test_gs,validate_gs

    def reader(self, csv_file):
        feat_nodes = []
        adj = []
        amp = []
        df = pd.read_csv(csv_file)
        

def hard_normalize(vec):
    return (vec - min(vec)) / (max(vec) - min(vec))


if __name__ == '__main__':
    dr = "/Users/andy/MainLand/Python/FeynGNN/data/test"
    gr = GraphSet()
    gr.creator(size=2)
    gr.saver(dir_root=dr)
    df = pd.read_csv(dr+'/test.csv')
    for string in df['nodes_feat']:
        string = string.strip('[]').strip('\n').split(' ')
        string = [float(x) for x in string if x]
        print(len(string))






