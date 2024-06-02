import torch 


class FeynGraph():
    def __init__(self,num_nodes,num_edges,amp):
        self.num_nodes = num_nodes
        self.num_edges = num_edges 
        #self.index_list = index_list 
        self.amp = amp 
        self.feat_nodes = []
        self.feat_edges = []

    def set_adj(self,adj):
        if isinstance(adj, torch.Tensor):
            if adj.size() == torch.Size([2,self.num_edges]):
                if adj.dtype == torch.int64:
                    self.index_list = adj
                else:
                    raise ValueError('Incorrect shape for adjacency matrix, expected to be [2, '+str(self.num_edges)+'], but got '+str(list(adj.shape))+'.')
            else:
                raise ValueError('Incorrect data type for adjacency matrix, expected to be int64, but got '+str(adj.dtype)+'.')
        else:
            raise TypeError('Adjacency matrix expected to be a torch tensor.')

    def set_feat_nodes(self, feat):
        if isinstance(feat, torch.Tensor):
            if feat.size() == torch.Size([self.num_nodes,6]):
                if feat.dtype == torch.float:
                    self.feat_nodes = feat
                else:
                    raise ValueError('Incorrect shape for nodes` feature matrix, expected to be ['+str(self.num_nodes)+', 6], but got '+str(list(feat.shape))+'.')
            else:
                raise ValueError('Incorrect data type for nodes` feature matrix, expected to be float, but got '+str(feat.dtype)+'.')
        else:
            raise TypeError('Nodes` feature matrix expected to be a torch tensor.')

    def set_feat_edges(self, feat):
        if isinstance(feat, torch.Tensor):
            if feat.size() == torch.Size([self.num_edges,2]):
                if feat.dtype == torch.float:
                    self.feat_edges = feat
                else:
                    raise ValueError('Incorrect data type for edges` feature matrix, expected to be int64, but got '+str(feat.dtype)+'.')
            else:
                raise ValueError('Incorrect shape for edges` feature matrix, expected to be ['+str(self.num_edges)+', 2], but got '+str(list(feat.shape))+'.')
        else:
            raise TypeError('Edges` feature matrix expected to be a torch tensor.')

    def set_amp(self, amp):
        if isinstance(amp, float):
            self.amp = amp 
        else:
            TypeError('Amplitude expected to be a float.')

    def get_feat_nodes(self):
        return self.feat_nodes

    def get_feat_edges(self):
        """    
        Abandoned for instance.
        """
        return self.feat_edges

    def get_adj(self):
        return self.adj 

    def get_amp(self):
        return self.amp