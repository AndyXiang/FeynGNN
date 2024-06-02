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


class GCNLayer(MessagePassing):
    def __init__(self,node_emb_dim=16,edge_emb_dim=4,aggr='add',dropout=0.2):
        super().__init__(aggr=aggr)
        self.msg = nn.Sequential(
            nn.Linear(2*node_emb_dim+edge_emb_dim, node_emb_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )
        for layer in self.msg:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
        self.upd = nn.Sequential(
            nn.Linear(2*node_emb_dim, node_emb_dim),
            nn.BatchNorm1d(5),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )
        for layer in self.upd:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

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
    def __init__(self, num_convs=4, node_feat_dim=6, edge_feat_dim=2,out_dim=1, node_emb_dim=16,edge_emb_dim=4,dropout=0.2):
        super().__init__()
        self.node_emb_dim = node_emb_dim
        self.node_emb = nn.Linear(node_feat_dim, node_emb_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.constant_(self.node_emb.bias, 0)

        self.edge_emb = nn.Linear(edge_feat_dim, edge_emb_dim)
        nn.init.xavier_uniform_(self.edge_emb.weight)
        nn.init.constant_(self.edge_emb.bias, 0)
        
        self.pool =  nn.Sequential(
            nn.Linear(5*node_emb_dim,node_emb_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
        )
        for layer in self.pool:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        self.MLP = nn.Sequential(
            nn.Linear(node_emb_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128,8),
            nn.AvgPool1d(8),
        )
        for layer in self.MLP:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)

        self.num_convs = num_convs
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            self.convs.append(GCNLayer(node_emb_dim=self.node_emb_dim, edge_emb_dim=edge_emb_dim,dropout=dropout))


    def forward(self, node_feat, edge_feat,edge_index):
        x = self.node_emb(node_feat)
        #x = node_feat
        edge_attr = self.edge_emb(edge_feat)
        for conv in self.convs:
            x = x + conv(x, edge_index, edge_attr=edge_attr)
        temp=[]
        for i in range(5):
            temp.append(x[:,i,:])
        x = self.pool(torch.cat(temp,dim=-1))
        x = self.MLP(x)
        return x


def Training(trainset,testset, lr=0.1,batch_size=20,num_epoch=20,loss_limit=0.01,loss_func=F.mse_loss,num_convs=2,node_emb_dim=16,edge_emb_dim=4,GPU=False):
    len_proc = len(trainset.proc_list)
    proc_list = trainset.proc_list
    edge_index_list = []
    for proc in proc_list:
        edge_index_list.append(pp.Index_List[proc])
    batch = {'node_feat': [None for i in range(len_proc)], 'edge_feat': [None for i in range(len_proc)],'amp': [None for i in range(len_proc)] }
    num_batch = int(trainset.size/batch_size)
    for i in range(len_proc):
        l = len(trainset.feat_edges[i][0])
        batch['node_feat'][i] = torch.cat([trainset.feat_nodes[i][batch_size*j:batch_size*(j+1),:,:] for j in range(num_batch)],dim=0).view(num_batch,batch_size,5,6)
        batch['edge_feat'][i] = torch.cat([trainset.feat_edges[i][batch_size*j:batch_size*(j+1),:,:] for j in range(num_batch)],dim=0).view(num_batch,batch_size,l,2)
        batch['amp'][i] = torch.cat([trainset.amp[i][batch_size*j:batch_size*(j+1)] for j in range(num_batch)],dim=0).view(num_batch,batch_size)
    model = GNNModel(num_convs=num_convs,node_emb_dim=node_emb_dim,edge_emb_dim=edge_emb_dim)
    if GPU:
        model.to('cuda')
        for i in range(len_proc):
            batch['node_feat'][i] = batch['node_feat'][i].to('cuda')
            batch['edge_feat'][i] = batch['edge_feat'][i].to('cuda')
            batch['amp'][i] = batch['amp'][i].to('cuda')
            testset.feat_nodes[i] = testset.feat_nodes[i].to('cuda')
            testset.feat_edges[i] = testset.feat_egdes[i].to('cuda')
            testset.amp[i] = testset.amp[i].to('cuda')
            edge_index_list[i].to('cuda')
    order = []
    for i in range(len_proc):
        for _ in range(num_batch):
            order.append(i)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    for epoch in range(num_epoch):
        rd.shuffle(order)
        ite = [0 for i in range(len_proc)]
        for t in tqdm(range(len_proc*num_batch),desc='epoch: '+str(epoch+1)):
            proc = order[t]
            out = model(batch['node_feat'][proc][ite[proc]], batch['edge_feat'][proc][ite[proc]], edge_index_list[proc])
            loss = loss_func(out.view(batch_size), batch['amp'][proc][ite[proc]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        loss = []
        for t in range(len_proc):
            out = model(testset.feat_nodes[t], testset.feat_edges[t], edge_index_list[t])
            loss.append(loss_func(out.view(len(out)), testset.amp[t]).item())
        print(loss)
        if loss < [loss_limit for i in range(len_proc)]:
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

def Figuring(model, proc_list,dir_root):
    ang = [1+i*0.03 for i in range(50)]
    for p in proc_list:
        index = pp.Index_List[p]
        l = len(pp.Index_List[p][0])
        for E in [300,800,1500,2000]:
            proc = getattr(pp, p)
            amp_theo = torch.tensor([proc(E, 1, 'electron', 'muon', ang[i]).get_amp() for i in range(50)],dtype=torch.float)
            amp_theo = dh.HardNormalize(amp_theo)
            feat_nodes = torch.cat([proc(E, 1, 'electron', 'muon', ang[i]).get_feat_nodes() for i in range(50)],dim=0).view(50,5,6)
            feat_edges = torch.cat([proc(E, 1, 'electron', 'muon', ang[i]).get_feat_edges() for i in range(50)],dim=0).view(50,l,2)
            amp_pred = model(feat_nodes, feat_edges, index).view(50).tolist()
            plt.scatter(ang, amp_theo, label='Thepratical Calculation',s=10)
            plt.scatter(ang, amp_pred, label='Model Predictions',s=10)
            plt.legend()
            plt.savefig(dir_root+str(p)+'_'+str(E)+'.png')
            plt.close()


if __name__ == '__main__':
    dr = '/Users/andy/MainLand/Python/FeynGNN/data/'
    gr = dh.GraphSet(proc_list=[])
    gr.reader(dir_root=dr)
    trainset, testset, valiset = gr.spliter()
    model = Training(trainset, testset,num_epoch=20,num_convs=2)
    Validating(model, valiset)
    Figuring(model, proc_list=trainset.proc_list, dir_root='/Users/andy/MainLand/Python/validatingFIG/')
