import PhysicsProcess as pp
from FeynGraph import FeynGraph
import numpy as np
import torch 
import pandas as pd
import os 
import random as rd


class GraphSet:
    def __init__(self,proc_list):
        self.proc_list = proc_list
        self.feat_nodes = [[] for i in range(len(self.proc_list))]
        self.feat_edges = [[] for i in range(len(self.proc_list))]
        self.amp = [[] for i in range(len(self.proc_list))]
        self.size = 0


    def creator(self, size:int=10000,energy_range:tuple=(400,2000),angular_range:tuple=(1,2.5),random_energy=False,seed=None):
        self.size = 4*size
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
                graph = proc(ECM[i],-1,'electron','muon',ang[i])
                self.feat_nodes[p].append(graph.get_feat_nodes())
                self.feat_edges[p].append(graph.get_feat_edges())
                self.amp[p].append(graph.get_amp())
                graph = proc(ECM[i],1,'muon','electron',ang[i])
                self.feat_nodes[p].append(graph.get_feat_nodes())
                self.feat_edges[p].append(graph.get_feat_edges())
                self.amp[p].append(graph.get_amp())
            l = len(self.feat_edges[p][0])
            self.feat_nodes[p] = TensorShuffle(torch.cat(self.feat_nodes[p],dim=0).view(4*size,5,7))
            self.feat_edges[p] = TensorShuffle(torch.cat(self.feat_edges[p],dim=0).view(4*size,l,2))
            self.amp[p] =TensorShuffle(torch.tensor(self.amp[p],dtype=torch.float))

    def saver(self, dir_root):
        for p in range(len(self.proc_list)):
            proc = self.proc_list[p]
            index = pp.Index_List[proc]
            filename = dir_root +proc +'_data.csv'
            datadic = {}
            for i in range(5):#5 is num_node
                datadic['mass'+str(i)] = self.feat_nodes[p][:,i,0]
                datadic['charge'+str(i)] = self.feat_nodes[p][:,i,1]
                datadic['spin'+str(i)] = self.feat_nodes[p][:,i,2]
                datadic['energy'+str(i)] = self.feat_nodes[p][:,i,3]
                datadic['momentum'+str(i)] = self.feat_nodes[p][:,i,4]
                datadic['angular'+str(i)] = self.feat_nodes[p][:,i,5]
                datadic['type'+str(i)] = self.feat_nodes[p][:,i,6]
            for i in range(len(self.feat_edges[p][0])):
                datadic['type'+str(index[0,i].item())+str(index[1,i].item())] = self.feat_edges[p][:,i,0]
                datadic['counts'+str(index[0,i].item())+str(index[1,i].item())] = self.feat_edges[p][:,i,1]
            datadic['amp'] = self.amp[p]
            pd.DataFrame(datadic).to_csv(filename,index=False)

    def spliter(self, ratio=(8,1,1)):
        train_gs, test_gs, validate_gs = GraphSet(self.proc_list),GraphSet(self.proc_list),GraphSet(self.proc_list)
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

    def reader(self,dir_root):
        proc_list = []
        file_list = []
        for root, dirs, filenames in os.walk(dir_root):
            for filename in filenames:
                file_list.append(os.path.join(root, filename))
                proc_list.append(os.path.join(root, filename).split('\\')[-1].split('_')[0])
        feat_nodes = []
        feat_edges = []
        amp = []
        for i in range(len(proc_list)):
            df = pd.read_csv(file_list[i])
            size = df.shape[0]
            temp = []
            for j in range(5):
                subset = df.iloc[:,7*j:7*j+7]
                temp.append(torch.tensor(subset.values,dtype=torch.float).view(size,1,7))
            feat_nodes.append(torch.cat(temp,dim=1))
            temp = []
            for j in range(len(pp.Index_List[proc_list[i]][0])):
                subset = df.iloc[:,35+2*j:37+2*j]
                temp.append(torch.tensor(subset.values,dtype=torch.int64).view(size,1,2))
            feat_edges.append(torch.cat(temp,dim=1).float())
            amp.append(HardNormalize(torch.tensor(df.iloc[:,-1].values,dtype=torch.float)))
        self.proc_list = proc_list
        self.amp = amp 
        self.feat_edges = feat_edges
        self.feat_nodes = feat_nodes
        self.size = len(amp[0])

def HardNormalize(vec):
    return (vec - min(vec)) / (max(vec) - min(vec))

def TensorShuffle(tensor, seed=100):
    torch.manual_seed(seed)
    index = torch.randperm(tensor.size(0))
    shuffled_tensor = tensor[index]
    return shuffled_tensor

if __name__ == '__main__':
    dr = "D:\\Python\\data\\"
    gr = GraphSet(proc_list=["PairAnnihilation","ColumbScattering"])
    gr.creator(size=3000,random_energy=True)
    gr.saver(dir_root=dr)





