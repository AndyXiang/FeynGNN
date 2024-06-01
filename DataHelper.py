import PhysicsProcess as pp
from FeynGraph import FeynGraph
import numpy as np
import torch 
import pandas as pd
import os 


class GraphSet:
    def __init__(self,proc_list):
        self.proc_list = proc_list
        self.feat_nodes = [[] for i in range(len(self.proc_list))]
        self.feat_edges = [[] for i in range(len(self.proc_list))]
        self.amp = [[] for i in range(len(self.proc_list))]


    def creator(self, size:int=10000,energy_range:tuple=(400,2000),angular_range:tuple=(1,2.5),random_energy=False,seed=None):
        ang = (angular_range[1]-angular_range[0])*np.random.random(size)+angular_range[0]
        if random_energy:
            np.random.seed(seed)
            ECM = (energy_range[1]-energy_range[0])*np.random.random(size)+energy_range[0]
        else:
            ECM = np.arange(start=energy_range[0],stop=energy_range[1],step=(energy_range[1]-energy_range[0])/size)
        for p in range(len(self.proc_list)):
            proc = getattr(pp, self.proc_list[p])
            for i in range(size):
                graph = proc(ECM[i],1,'electron','muon',ang[i])
                self.feat_nodes[p].append(graph.get_feat_nodes())
                self.feat_edges[p].append(graph.get_feat_edges())
                self.amp[p].append(graph.get_amp())
                graph = proc(ECM[i],-1,'muon','electron',ang[i])
                self.feat_nodes[p].append(graph.get_feat_nodes())
                self.feat_edges[p].append(graph.get_feat_edges())
                self.amp[p].append(graph.get_amp())
            l = len(self.feat_edges[p][0])
            self.feat_nodes[p] = torch.cat(self.feat_nodes[p],dim=0).view(2*size,5,6)
            self.feat_edges[p] = torch.cat(self.feat_edges[p],dim=0).view(2*size,l,2)
            self.amp[p] = torch.tensor(self.amp[p],dtype=torch.float)

    def saver(self, dir_root):
        for p in range(len(self.proc_list)):
            proc = self.proc_list[p]
            index = pp.Index_List[proc]
            filename = dir_root + proc +'_data.csv'
            datadic = {}
            for i in range(5):#5 is num_node
                datadic['mass'+str(i)] = self.feat_nodes[p][:,i,0]
                datadic['charge'+str(i)] = self.feat_nodes[p][:,i,1]
                datadic['spin'+str(i)] = self.feat_nodes[p][:,i,2]
                datadic['energy'+str(i)] = self.feat_nodes[p][:,i,3]
                datadic['momentum'+str(i)] = self.feat_nodes[p][:,i,4]
                datadic['angular'+str(i)] = self.feat_nodes[p][:,i,5]
            for i in range(len(self.feat_edges[p][0])):
                datadic['type'+str(index[0,i].item())+str(index[1,i].item())] = self.feat_edges[p][:,i,0]
                datadic['counts'+str(index[0,i].item())+str(index[1,i].item())] = self.feat_edges[p][:,i,1]
            datadic['amp'] = self.amp[p]
            pd.DataFrame(datadic).to_csv(filename,index=False)

def reader(dir_root):
    proc_list = []
    file_list = []
    for root, dirs, filenames in os.walk(dir_root):
        for filename in filenames:
            file_list.append(os.path.join(root, filename))
            proc_list.append(os.path.join(root, filename).split('/')[-1].split('_')[0])
    feat_nodes = []
    feat_edges = []
    amp = []
    for i in range(len(proc_list)):
        df = pd.read_csv(file_list[i])
        size = df.shape[0]
        subset = df.iloc[:,0:6]
        temp = []
        for j in range(5):
            subset = df.iloc[:,6*j:6*j+6]
            temp.append(torch.tensor(subset.values,dtype=torch.float).view(size,1,6))
        feat_nodes.append(torch.cat(temp,dim=1))
        for j in range(len(pp.Index_List[proc_list[i]])):
            print(1)



if __name__ == '__main__':
    dr = '/Users/andy/MainLand/Python/GNN for QED/datastore/test/'
    gs = GraphSet(proc_list=pp.Proc_List)
    gs.creator(size=10)
    gs.saver(dr)
    reader(dr)
    


