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
        MLP_params:list, act_func,num_epoch:int, batch_size:int, loss_func, lr:tuple, heads
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
        self.heads = heads

    def save_hyper(self,dr):
        text = ("node_emb_dim = "+ str(self.node_emb_dim)+'\n'
            +"num_convs = "+ str(self.num_convs)+'\n'
            +"pool_dim = "+ str(self.pool_dim)+'\n'
            +"act_func = "+str(self.act_func)+'\n'
            +"num_epoch = "+str(self.num_epoch)+'\n'
            +"batch_size = "+str(self.batch_size)+'\n'
            +"loss_func = "+str(self.loss_func)+'\n'
            +"lr = "+str(self.lr)+'\n'
            +"heads = "+str(self.heads)+'\n'
        )
        with open(dr+'/hyperparams.txt','w') as f:
            f.write(text)


class GATLayer(nn.Module):
    def __init__(self, heads=4, node_feat_dim=7, node_emb_dim=16):
        super().__init__()
        self.heads = heads
        dim = int(node_emb_dim/heads)
        self.W = nn.ModuleList()
        for i in range(heads):
            self.W.append(
                nn.Sequential(
                nn.Linear(node_feat_dim, dim),
                nn.Dropout(0.1),
                )
            )

        self.Q = nn.Sequential(
            nn.Linear(2*dim, 1),
            nn.Dropout(0.1),
        )

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
        out = torch.cat(tuple(temp), dim=2)

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
        self.convs.append(GATLayer(node_feat_dim=node_feat_dim, node_emb_dim=hyperparam.node_emb_dim, heads=hyperparam.heads))
        for i in range(hyperparam.num_convs-1):
            self.convs.append(GATLayer(node_feat_dim=hyperparam.node_emb_dim, node_emb_dim=hyperparam.node_emb_dim, heads=hyperparam.heads))

    def forward(self, x, adj):
        for conv in self.convs:
            x = conv(x, adj)
        x = self.poolin(torch.sum(x, dim=1))
        for layer in self.MLP:
            x = layer(x)
        return torch.exp(x)


def r2(p_value, t_value):
    m = torch.mean(p_value)
    return torch.sum(torch.square(p_value - m)) / torch.sum(torch.square(t_value - m))

def Training(hyperparam:HyperParams, loss_limit, GPU, fig_root):
    proc_list = pp.PROC_LIST
    len_proc = len(proc_list)
    # generate training dataset and testing dataset
    train_set = dh.GraphSet()
    test_set = dh.GraphSet()
    validate_set = dh.GraphSet()
    train_set.reader('data/train40000')
    test_set.reader('data/test40')
    validate_set.reader('data/validate40')
    # batchlize the training dataset
    num_batch = int(train_set.size/hyperparam.batch_size)
    batch = dh.GraphSet()
    batch.proc_list = train_set.proc_list
    for i in range(len_proc):
        batch.dataset[i]['nodes_feat'] = torch.tensor(np.reshape(
            train_set.dataset[i]['nodes_feat'], 
            (num_batch, hyperparam.batch_size, train_set.dataset[i]['nodes_feat'].shape[1], 7)
        ),dtype=torch.float32)
        batch.dataset[i]['amp'] = torch.tensor(np.reshape(
            train_set.dataset[i]['amp'], 
            (num_batch, hyperparam.batch_size)
        ),dtype=torch.float32)
        test_set.dataset[i]['nodes_feat'] = torch.tensor(test_set.dataset[i]['nodes_feat'], dtype=torch.float32)
        test_set.dataset[i]['amp'] = torch.tensor(test_set.dataset[i]['amp'], dtype=torch.float32)
        validate_set.dataset[i]['nodes_feat'] = torch.tensor(validate_set.dataset[i]['nodes_feat'], dtype=torch.float32)
        validate_set.dataset[i]['amp'] = torch.tensor(validate_set.dataset[i]['amp'], dtype=torch.float32)
    # initiate the model and move the tensors to gpu when needed
    model = GNNModel(hyperparam=hyperparam)
    weights = model.state_dict()
    for name, param in weights.items():
        if 'weight' in name:
            nn.init.xavier_normal_(param.unsqueeze(0))
            param.squeeze()
        elif 'bias' in name:
            param.data.fill_(0)
    loss_func = hyperparam.loss_func
    if GPU:
        model.to('cuda')
        r2.to('cuda')
        for i in range(len_proc):
            batch.dataset[i]['nodes_feat'] = batch.dataset[i]['nodes_feat'].to('cuda')
            batch.dataset[i]['amp'] = batch.dataset[i]['amp'].to('cuda')
            test_set.dataset[i]['nodes_feat'] = test_set.dataset[i]['nodes_feat'].to('cuda')
            test_set.dataset[i]['amp'] = test_set.dataset[i]['amp'].to('cuda')
            pp.ADJ_LIST[pp.PROC_LIST[i]].to('cuda')
    # the random order for training on different processes
    order = []
    for i in range(len_proc):
        for _ in range(num_batch):
            order.append(i)
    # train the model
    optimizer = torch.optim.RMSprop(model.parameters(), lr=hyperparam.lr)
    for epoch in range(hyperparam.num_epoch):
        model.train()
        rd.shuffle(order)
        ite = [0 for _ in range(len_proc)]
        for t in tqdm(range(len_proc*num_batch) ,desc='epoch: '+str(epoch+1)):
            proc = order[t]
            out = model(batch.dataset[proc]['nodes_feat'][ite[proc]],  pp.ADJ_LIST[pp.PROC_LIST[proc]])
            loss = loss_func(out.view(hyperparam.batch_size), batch.dataset[proc]['amp'][ite[proc]])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ite[proc]+=1
        model.eval()
        loss = []
        for t in range(len_proc):
            out = model(test_set.dataset[t]['nodes_feat'], pp.ADJ_LIST[pp.PROC_LIST[t]])
            loss.append(loss_func(out.view(40), test_set.dataset[t]['amp']).item())
            text = proc_list[t]+': loss =  '+str(loss[t])
            print(text)
        if sum(loss) < len_proc*loss_limit:
            break # break the training when loss is low enough
    print("Model Training Completed.")
    model.to('cpu')
    for i in range(len_proc):
        pred = model(validate_set.dataset[i]['nodes_feat'], pp.ADJ_LIST[pp.PROC_LIST[i]]).view(40)
        
        text = proc_list[i] + ' RSquare: ' + str(r2(pred, validate_set.dataset[i]['amp']).item())
        print(text)
        x = [j for j in range(40)]
        plt.scatter(x, validate_set.dataset[i]['amp'].tolist(), label='Theoretical Calculation',s=10)
        plt.scatter(x, pred.tolist(), label='Model Predictions',s=10)
        plt.legend()
        plt.savefig(dir_root+'/'+proc_list[i]+'.png')
        plt.close()
    return model


if __name__ == '__main__':
    hyperparam = HyperParams(
        node_emb_dim=16,
        num_convs=2,
        pool_dim=128,
        MLP_params=[(128, 256),(256,256),(256, 128)],
        act_func=nn.LeakyReLU(negative_slope=0.01, inplace=True),
        num_epoch=100,
        batch_size=40,
        loss_func=F.l1_loss,
        lr=0.01,
        heads=1
    )
    model = Training(hyperparam,loss_limit=1e-7,GPU=False,fig_root='model/train40000/fig')
    torch.save(model.state_dict(),'model/train40000/model.pt')
    hyperparam.save_hyper(dr="model/train40000")
