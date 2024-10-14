import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torchmetrics import R2Score
from tqdm import tqdm 
import matplotlib.pyplot as plt
import PhysicsProcess as pp
import DataHelper as dh #data
import random as rd
import pandas as pd


class HyperParams:
    def __init__(
        self,node_emb_dim:int, num_convs:int, pool_dim:int, 
        MLP_params:list, act_func,num_epoch:int, batch_size:int, loss_func, lr:tuple
        ):
        self.node_emb_dim = node_emb_dim
        self.num_convs = num_convs
        self.pool_dim = pool_dim
        self.MLP_params = MLP_params
        self.act_func = act_func
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.lr = lr

    def save_hyper(self,dr):
        text = ("node_emb_dim = "+ str(self.node_emb_dim)+'\n'
            +"num_convs = "+ str(self.num_convs)+'\n'
            +"pool_dim = "+ str(self.pool_dim)+'\n'
            +"act_func = "+str(self.act_func)+'\n'
            +"num_epoch = "+str(self.num_epoch)+'\n'
            +"batch_size = "+str(self.batch_size)+'\n'
            +"loss_func = "+str(self.loss_func)+'\n'
            +"lr = "+str(self.lr)+'\n'
        )
        with open(dr+'hyperparams.txt','w') as f:
            f.write(text)

class GATLayer(nn.Module):
    def __init__(self, heads=4, node_feat_dim=7, node_emb_dim=16):
        super().__init__()
        self.heads = heads
        temp = int(node_emb_dim/4)
        self.W = nn.ModuleList()
        for i in range(heads):
            self.W.append(nn.Linear(node_feat_dim, temp))

        self.Q = nn.Linear(2*temp, 1)


    def forward(self, h, adj):
        temp = []
        for i in range(self.heads):
            W = self.W[i]
            x1 = W(h).unsqueeze(2)
            num = x1.size()[1]
            x2 = torch.cat([x1 for i in range(num)], dim=2)
            x3 = self.Q(torch.cat((x2, x2), dim=3)).squeeze()
            a = F.softmax(F.leaky_relu(adj*x3), dim=2)
            temp.append(a @ W(h))
        out = F.leaky_relu(torch.cat(tuple(temp), dim=2))

        return out
    

class GNNModel(nn.Module):
    def __init__(self,  hyperparam:HyperParams, node_feat_dim=7):
        super().__init__()
        self.act_func = hyperparam.act_func
        
        # Linear pooling of graph
        self.poolin =  nn.Sequential(
            nn.Linear(hyperparam.node_emb_dim, hyperparam.pool_dim),
            nn.BatchNorm1d(hyperparam.pool_dim),
            self.act_func,
            nn.Dropout(0.1),
        )

        # Multi-layer Percptron settings
        self.MLP = nn.ModuleList()
        for t in hyperparam.MLP_params:
            self.MLP.append(
                nn.Sequential(
                    nn.Linear(t[0],t[1]),
                    nn.BatchNorm1d(t[1]),
                    self.act_func,
                    nn.Dropout(0.1),
                )
            )
        self.MLP.append(nn.AvgPool1d(hyperparam.MLP_params[-1][1]))

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GATLayer(node_feat_dim=node_feat_dim, node_emb_dim=hyperparam.node_emb_dim))
        self.convs.append(GATLayer(node_feat_dim=hyperparam.node_emb_dim, node_emb_dim=hyperparam.node_emb_dim))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        x = self.poolin(torch.sum(x, dim=1))
        for layer in self.MLP:
            x = layer(x)
        return x


def Training(size, hyperparam:HyperParams, loss_limit=0.01, GPU=False, seed=1000):
    len_proc = 3
    proc_list = pp.PROC_LIST
    trainset = {
        'PairAnnihilation':{'nodes_feat':[], 'adj':[], 'amp':[]},
        'CoulombScattering':{'nodes_feat':[], 'adj':[], 'amp':[]},
        'ComptonScattering':{'nodes_feat':[], 'adj':[], 'amp':[]}
    }
    for t in range(3):
        proc = getattr(pp, proc_list[t])
        for i in range(size):
            adj, node_feat, amp = proc('electron', 1, seed=seed)
            trainset[proc_list[t]]['adj'].append(adj)
            trainset[proc_list[t]]['nodes_feat'].append(node_feat)
            trainset[proc_list[t]]['amp'].append(amp)
            adj, node_feat, amp = proc('electron', -1, seed=seed)
            trainset[proc_list[t]]['adj'].append(adj)
            trainset[proc_list[t]]['nodes_feat'].append(node_feat)
            trainset[proc_list[t]]['amp'].append(amp)
            adj, node_feat, amp = proc('muon', 1, seed=seed)
            trainset[proc_list[t]]['adj'].append(adj)
            trainset[proc_list[t]]['nodes_feat'].append(node_feat)
            trainset[proc_list[t]]['amp'].append(amp)
            adj, node_feat, amp = proc('muon', -1, seed=seed)
            trainset[proc_list[t]]['adj'].append(adj)
            trainset[proc_list[t]]['nodes_feat'].append(node_feat)
            trainset[proc_list[t]]['amp'].append(amp)
        trainset[proc_list[t]]['adj'] = torch.tensor(np.array(trainset[proc_list[t]]['adj']), dtype=torch.float32)
        trainset[proc_list[t]]['nodes_feat'] = torch.tensor(np.array(trainset[proc_list[t]]['nodes_feat']), dtype=torch.float32)
        trainset[proc_list[t]]['amp'] = torch.tensor(np.array(trainset[proc_list[t]]['amp']), dtype=torch.float32)

    testset = {
        'PairAnnihilation':{'nodes_feat':[], 'adj':[], 'amp':[]},
        'CoulombScattering':{'nodes_feat':[], 'adj':[], 'amp':[]},
        'ComptonScattering':{'nodes_feat':[], 'adj':[], 'amp':[]}
    }
    for t in range(3):
        proc = getattr(pp, proc_list[t])
        for i in range(100):
            adj, node_feat, amp = proc('electron', 1, seed=seed)
            testset[proc_list[t]]['adj'].append(adj)
            testset[proc_list[t]]['nodes_feat'].append(node_feat)
            testset[proc_list[t]]['amp'].append(amp)
            adj, node_feat, amp = proc('electron', -1, seed=seed)
            testset[proc_list[t]]['adj'].append(adj)
            testset[proc_list[t]]['nodes_feat'].append(node_feat)
            testset[proc_list[t]]['amp'].append(amp)
            adj, node_feat, amp = proc('muon', 1, seed=seed)
            testset[proc_list[t]]['adj'].append(adj)
            testset[proc_list[t]]['nodes_feat'].append(node_feat)
            testset[proc_list[t]]['amp'].append(amp)
            adj, node_feat, amp = proc('muon', -1, seed=seed)
            testset[proc_list[t]]['adj'].append(adj)
            testset[proc_list[t]]['nodes_feat'].append(node_feat)
            testset[proc_list[t]]['amp'].append(amp)
        testset[proc_list[t]]['adj'] = torch.tensor(np.array(testset[proc_list[t]]['adj']), dtype=torch.float32)
        testset[proc_list[t]]['nodes_feat'] = torch.tensor(np.array(testset[proc_list[t]]['nodes_feat']), dtype=torch.float32)
        testset[proc_list[t]]['amp'] = torch.tensor(np.array(testset[proc_list[t]]['amp']), dtype=torch.float32)

    num_batch = int(4*size/hyperparam.batch_size)
    batch = {'nodes_feat':[None for i in range(len_proc)], 'adj':[None for i in range(len_proc)], 'amp':[None for i in range(len_proc)]}
    for i in range(len_proc):
        batch['nodes_feat'][i] = torch.cat(
            [trainset[proc_list[i]]['nodes_feat'][hyperparam.batch_size*j:hyperparam.batch_size*(j+1),:,:].unsqueeze(0) for j in range(num_batch)]
        ,dim=0)
        batch['adj'][i] = torch.cat(
            [trainset[proc_list[i]]['adj'][hyperparam.batch_size*j:hyperparam.batch_size*(j+1),:,:].unsqueeze(0) for j in range(num_batch)]
        ,dim=0)
        batch['amp'][i] = torch.cat(
            [trainset[proc_list[i]]['amp'][hyperparam.batch_size*j:hyperparam.batch_size*(j+1)].unsqueeze(0) for j in range(num_batch)]
        ,dim=0)
    model = GNNModel(hyperparam=hyperparam)
    loss_func = hyperparam.loss_func
    if GPU:
        model.to('cuda')
        for i in range(len_proc):
            batch['nodes_feat'][i] = batch['nodes_feat'][i].to('cuda')
            batch['adj'][i] = batch['adj'][i].to('cuda')
            batch['amp'][i] = batch['amp'][i].to('cuda')
    order = []
    for i in range(len_proc):
        for _ in range(num_batch):
            order.append(i)
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1)
    for epoch in range(hyperparam.num_epoch):
        lr_temp = hyperparam.lr[0]-(hyperparam.lr[0]-hyperparam.lr[1])*epoch/hyperparam.num_epoch
        for para in optimizer.param_groups:
            para['lr'] = lr_temp
        rd.shuffle(order)
        ite = [0 for _ in range(len_proc)]
        for t in tqdm(range(len_proc*num_batch) ,desc='epoch: '+str(epoch+1)):
            proc = order[t]
            out = model(
                batch['nodes_feat'][proc][ite[proc]],  batch['adj'][proc][ite[proc]]
            )
            loss = loss_func(out.view(hyperparam.batch_size), batch['amp'][proc][ite[proc]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ite[proc]+=1
        model.eval()
        loss = []
        for t in range(len_proc):
            out = model(testset[proc_list[t]]['nodes_feat'], testset[proc_list[t]]['adj'])
            out = out.view(len(out))
            loss.append(loss_func(out, testset[proc_list[t]]['amp']).item())
            text = proc_list[t]+' mse loss: '+str(loss[t])
            print(text)
        if loss < [loss_limit for _ in range(len_proc)]:
            break
    print("Model Training Completed.")
    return model

def Validating(model, valiset):
    r2 = R2Score()
    model.to('cpu')
    len_proc = len(valiset.proc_list)
    proc_list = valiset.proc_list
    model.eval()
    for i in range(len_proc):
        pred = model(valiset.feat_nodes[i],  pp.Index_List[proc_list[i]]).view(valiset.size)
        text = proc_list[i] + ' RSquare: '+str(r2(pred, valiset.amp[i]).item())
        print(text)

def Figuring(model, proc_list,dir_root,angular_range=(0.5,2)):
    ang = [angular_range[0]+i*(angular_range[1]-angular_range[0])/50 for i in range(50)]
    for p in proc_list:
        index = pp.Index_List[p]
        l = len(pp.Index_List[p][0])
        proc = getattr(pp, p)
        for E in [500,1000,1500,2000]: 
            amp_theo = torch.tensor(
                [proc(E, 1, 'electron', 'muon', ang[i]).get_amp() for i in range(50)],dtype=torch.float
            )
            amp_theo = dh.hard_normalize(amp_theo)
            feat_nodes = torch.cat(
                [proc(E, 1, 'electron', 'muon', ang[i]).get_feat_nodes() for i in range(50)],dim=0
            ).view(50,5,7)
            model.eval()
            amp_pred = model(feat_nodes, index).view(50).tolist()
            plt.scatter(ang, amp_theo, label='Thepratical Calculation',s=10)
            plt.scatter(ang, amp_pred, label='Model Predictions',s=10)
            plt.legend()
            plt.savefig(dir_root+str(p)+'_'+str(E)+'.png')
            plt.close()


if __name__ == '__main__':
    hyperparam = HyperParams(
        node_emb_dim=32,
        num_convs=2,
        pool_dim=512,
        MLP_params=[(512,1024),(1024,1024),(1024,1024),(1024,512),(512,32)],
        act_func=nn.LeakyReLU(inplace=True),
        num_epoch=40,
        batch_size=80,
        loss_func=F.mse_loss,
        lr=(0.02,0.002),
    )
    model = Training(10000,hyperparam,loss_limit=1e-7,GPU=False)
    torch.save(model.state_dict(),'/Users/andy/MainLand/Python/FeynGNN/model/test/GNNmodel.pt')
    hyperparam.save_hyper(dr="/Users/andy/MainLand/Python/FeynGNN/model/test/")
