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
        self.dataset = [{'nodes_feat':[], 'amp':[]} for _ in range(len(proc_list))]
        self.size = 0

    def creator(self, size:int=10000):
        # The real size of each process is 4*size
        self.size = 4*size
        for t in range(len(self.proc_list)):
            proc = getattr(pp, self.proc_list[t])
            for i in range(size):
                for particle in iter(['electron', 'muon']):
                    for charge in iter([1,-1]):
                        nodes_feat, amp = proc(particle, charge)
                        self.dataset[t]['nodes_feat'].append(nodes_feat)
                        self.dataset[t]['amp'].append(amp)
            self.dataset[t]['nodes_feat'] = np.array(self.dataset[t]['nodes_feat'])
            self.dataset[t]['amp'] = np.array(self.dataset[t]['amp'])

    def clearer(self, amp_limit=1000):
        # clear data that has abnormal amplitude
        for t in range(len(self.proc_list)):
            for i in range(self.size):
                while self.dataset[t]['amp'][i] > amp_limit:
                    proc = getattr(pp, self.proc_list[t])
                    nodes_feat, amp = proc('electron', 1)
                    self.dataset[t]['nodes_feat'][i] = nodes_feat
                    self.dataset[t]['amp'][i] = amp 

    # the following three methods need to be written
    def saver(self, dir_root):
        for i in range(len(self.proc_list)):
            folder_root = dir_root + '/' + pp.PROC_LIST[i]
            if not os.path.exists(folder_root):
                os.mkdir(folder_root)
            num_nodes = pp.ADJ_LIST[pp.PROC_LIST[i]].shape[0]
            np.savetxt(folder_root+'/feat.txt', self.dataset[i]['nodes_feat'].reshape(self.size, num_nodes*7))
            np.savetxt(folder_root+'/amp.txt', self.dataset[i]['amp'])

    def reader(self, dir_root):
        for i in range(len(self.proc_list)):
            folder_root = dir_root + '/' + pp.PROC_LIST[i]
            num_nodes = pp.ADJ_LIST[pp.PROC_LIST[i]].shape[0]
            feat = np.loadtxt(folder_root+'/feat.txt')
            amp = np.loadtxt(folder_root+'/amp.txt')
            self.size = amp.shape[0]
            self.dataset[i]['nodes_feat'] = feat.reshape(self.size, num_nodes, 7)
            self.dataset[i]['amp'] = amp 

def hard_normalize(vec):
    return (vec - min(vec)) / (max(vec) - min(vec))


if __name__ == '__main__':
    ds = GraphSet()
    ds.creator(size=10000)
    ds.clearer(amp_limit=100)
    for i in range(len(pp.PROC_LIST)):
        ds.dataset[i]['amp'] = hard_normalize(ds.dataset[i]['amp'])
    ds.saver('data/train40000')
    test = GraphSet()
    test.creator(size=10)
    test.clearer(amp_limit=100)
    for i in range(len(pp.PROC_LIST)):
        test.dataset[i]['amp'] = hard_normalize(test.dataset[i]['amp'])
    test.saver('data/test40')
    validate = GraphSet()
    validate.creator(size=10)
    validate.clearer(amp_limit=100)
    for i in range(len(pp.PROC_LIST)):
        validate.dataset[i]['amp'] = hard_normalize(validate.dataset[i]['amp'])
    validate.saver('data/validate40')





