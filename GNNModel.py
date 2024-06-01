import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing
from torchmetrics import R2Score
from tqdm import tqdm 
import matplotlib as plt
from PhysicsProcess import Index_List
import DataHelper as dh


class GCNLayer(MessagePassing):
    def __init__(self,node_emb_dim=16,edge_emb_dim=4,aggr='add',dropout=0.2):
        super().__init__(aggr=aggr)
        self.msg = nn.Sequential(
            nn.Linear(2*node_emb_dim+edge_emb_dim, node_emb_dim),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(node_emb_dim, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, node_emb_dim),
            nn.LeakyReLU(inplace=True),
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
            nn.Linear(node_emb_dim, 128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, node_emb_dim),
            nn.LeakyReLU(inplace=True),
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
        return torch_scatter.scatter(inputs, index, dim=self.node_dim, reduce=self.aggr)

    def update(self, aggr_out, h):
        upd = torch.cat([h, aggr_out],dim=-1)
        return self.upd(upd)

class GNNModel(nn.Module):
    def __init__(self, num_convs=4, node_feat_dim=7, edge_feat_dim=2,out_dim=1, node_emb_dim=16,edge_emb_dim=4,dropout=0.2):
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


def Training(graphset, batch_size=20,num_epoch=20,loss_limit=0.01,loss_func=F.mse_loss,num_convs=2,node_emb_dim=16,edge_emb_dim=4):
    batch = {'node_feat':[], 'edge_feat'::[[] for i in range(graphset.proc_list)], }