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
import DataHelper as dh
import random as rd
import pandas as pd


class HyperParams:
    def __init__(
        self,node_emb_dim:int, edge_emb_dim:int, num_convs:int, pool_dim:int, 
        MLP_params:list, act_func,num_epoch:int, batch_size:int, loss_func, lr:tuple, aggr:str
        ):
        self.node_emb_dim = node_emb_dim
        self.edge_emb_dim = edge_emb_dim
        self.num_convs = num_convs
        self.pool_dim = pool_dim
        self.MLP_params = MLP_params
        self.act_func = act_func
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.lr = lr
        self.aggr = aggr

    def save_hyper(self,dr):
        text = ("node_emb_dim = "+ str(self.node_emb_dim)+'\n'
            +"edge_emb_dim = "+str(self.edge_emb_dim)+'\n'
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

class MPLayer(MessagePassing):
    def __init__(self,node_feat_dim=16,edge_feat_dim=4 ,aggr='add'):
        super().__init__(aggr=aggr)

        self.msg = nn.Sequential(
            nn.Linear(2*node_feat_dim+edge_feat_dim, node_feat_dim),
            #nn.BatchNorm1d(index_len), 
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(node_feat_dim,node_feat_dim),
            #nn.BatchNorm1d(index_len),
            nn.LeakyReLU(inplace=True),
        )

        self.upd = nn.Sequential(
            nn.Linear(2*node_feat_dim, node_feat_dim),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(node_feat_dim,node_feat_dim),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, h, edge_index, edge_attr):
        out = self.propagate(edge_index, h=h, edge_attr=edge_attr)
        return out

    def message(self,h_i, h_j, edge_attr):
        msg = torch.cat([h_i, h_j, edge_attr], dim=-1)
        return self.msg(msg)

    def aggregate(self, inputs, index):
        return scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        upd = torch.cat([h, aggr_out],dim=-1)
        return self.upd(upd)

class GNNModel(nn.Module):
    def __init__(self,  hyperparam:HyperParams,node_feat_dim=7, edge_feat_dim=2):
        super().__init__()
        self.act_func = hyperparam.act_func
        # Linear embbeding of node features and edge features
        self.node_emb = nn.Linear(node_feat_dim, hyperparam.node_emb_dim)
        self.edge_emb = nn.Linear(edge_feat_dim, hyperparam.edge_emb_dim)
        
        # Linear pooling of graph
        self.poolin =  nn.Sequential(
            nn.Linear(5*hyperparam.node_emb_dim, hyperparam.pool_dim),
            nn.BatchNorm1d(hyperparam.pool_dim),
            self.act_func,
            nn.Dropout(0.1),
        )

        # Multi-layer Percptron settings
        self.MLP = nn.ModuleList()
        for t in hyperparam.MLP_paras:
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
        for i in range(hyperparam.num_convs):
            self.convs.append(MPLayer(
                node_feat_dim=hyperparam.node_emb_dim,edge_feat_dim=hyperparam.edge_emb_dim,aggr=hyperparam.aggr
                ))

    def forward(self, node_feat, edge_feat,edge_index):
        x = self.node_emb(node_feat)
        edge_attr = self.edge_emb(edge_feat)
        for conv in self.convs:
            x = x + conv(x, edge_index, edge_attr=edge_attr)
        x = self.poolin(torch.cat([x[:,i,:] for i in range(5)],dim=-1))
        for layer in self.MLP:
            x = layer(x)
        return x


def Training(trainset:dh.GraphSet,testset:dh.GraphSet,hyperparam:HyperParams,loss_limit=0.01,GPU=False):
    len_proc = len(trainset.proc_list)
    proc_list = trainset.proc_list
    edge_index_list = [pp.Index_List[proc] for proc in proc_list]
    batch = {
        'node_feat': [None for i in range(len_proc)], 
        'edge_feat': [None for i in range(len_proc)],
        'amp': [None for i in range(len_proc)] 
    }
    num_batch = int(trainset.size/hyperparam.batch_size)
    for i in range(len_proc):
        l = len(trainset.feat_edges[i][0])
        batch['node_feat'][i] = torch.cat(
            [trainset.feat_nodes[i][hyperparam.batch_size*j:hyperparam.batch_size*(j+1),:,:] for j in range(num_batch)],dim=0
            ).view(num_batch,hyperparam.batch_size,5,7)
        batch['edge_feat'][i] = torch.cat(
            [trainset.feat_edges[i][hyperparam.batch_size*j:hyperparam.batch_size*(j+1),:,:] for j in range(num_batch)],dim=0
            ).view(num_batch,hyperparam.batch_size,l,2)
        batch['amp'][i] = torch.cat(
            [trainset.amp[i][hyperparam.batch_size*j:hyperparam.batch_size*(j+1)] for j in range(num_batch)],dim=0
            ).view(num_batch,hyperparam.batch_size)
    model = GNNModel(hyperparam=hyperparam)
    loss_func = hyperparam.loss_func
    if GPU:
        model.to('cuda')
        for i in range(len_proc):
            batch['node_feat'][i] = batch['node_feat'][i].to('cuda')
            batch['edge_feat'][i] = batch['edge_feat'][i].to('cuda')
            batch['amp'][i] = batch['amp'][i].to('cuda')
            testset.feat_nodes[i] = testset.feat_nodes[i].to('cuda')
            testset.feat_edges[i] = testset.feat_edges[i].to('cuda')
            testset.amp[i] = testset.amp[i].to('cuda')
            edge_index_list[i]=edge_index_list[i].to('cuda')
    order = []
    for i in range(len_proc):
        for _ in range(num_batch):
            order.append(i)
    model.train()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1)
    for epoch in range(hyperpara.num_epoch):
        lr_temp = hyperparam.lr[0]-(hyperparam.lr[0]-hyperparam.lr[1])*epoch/hyperparam.num_epoch
        for para in optimizer.param_groups:
            para['lr'] = lr_temp
        rd.shuffle(order)
        ite = [0 for _ in range(len_proc)]
        for t in tqdm(range(len_proc*num_batch),desc='epoch: '+str(epoch+1)):
            proc = order[t]
            out = model(
                batch['node_feat'][proc][ite[proc]], batch['edge_feat'][proc][ite[proc]], edge_index_list[proc]
            )
            loss = loss_func(out.view(hyperparam.batch_size), batch['amp'][proc][ite[proc]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ite[proc]+=1
        model.eval()
        loss = []
        for t in range(len_proc):
            out = model(testset.feat_nodes[t], testset.feat_edges[t], edge_index_list[t])
            out = out.view(len(out))
            loss.append(loss_func(out, testset.amp[t]).item())
            text = proc_list[t]+'mse loss: '+str(loss[t])
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
        pred = model(valiset.feat_nodes[i], valiset.feat_edges[i], pp.Index_List[proc_list[i]]).view(valiset.size)
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
            feat_edges = torch.cat(
                [proc(E, 1, 'electron', 'muon', ang[i]).get_feat_edges() for i in range(50)],dim=0
            ).view(50,l,2)
            model.eval()
            amp_pred = model(feat_nodes, feat_edges, index).view(50).tolist()
            plt.scatter(ang, amp_theo, label='Thepratical Calculation',s=10)
            plt.scatter(ang, amp_pred, label='Model Predictions',s=10)
            plt.legend()
            plt.savefig(dir_root+str(p)+'_'+str(E)+'.png')
            plt.close()


if __name__ == '__main__':
    dr = "D:\\Python\\FeynGNN\\data\\4proc_full_size=20000(random_energy,ang=(0.5,2))\\"
    gr = dh.GraphSet(proc_list=[])
    gr.reader(dir_root=dr)
    trainset, testset, valiset = gr.spliter()
    hyperparam = HyperPara(
        node_emb_dim=32,
        edge_emb_dim=4,
        num_convs=2,
        pool_dim=512,
        MLP_paras=[(512,1024),(1024,1024),(1024,1024),(1024,512),(512,32)],
        act_func=nn.LeakyReLU(inplace=True),
        num_epoch=20,
        batch_size=40,
        loss_func=F.mse_loss,
        lr=(0.02,0.002),
        aggr='add'
    )
    model = Training(trainset, testset,hyperparam,loss_limit=5e-6,GPU=True)
    Validating(model, valiset)
    Figuring(model, proc_list=trainset.proc_list, dir_root="D:\\Python\\FeynGNN\\model\\4proc_full_size=20000\\validatingFIG\\")
    torch.save(model.state_dict(),'D:\\Python\\FeynGNN\\model\\4proc_full_size=20000\\GNNmodel.pt')
    hyperparam.saveHyper(dr="D:\\Python\\FeynGNN\\model\\4proc_full_size=20000\\")
