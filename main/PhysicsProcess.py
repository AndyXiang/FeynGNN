import torch
import numpy as np
from FeynGraph import FeynGraph
from math import cos

mass = {'electron': 0.51099895, 'muon':105.6583775, 'photon':0}

Proc_List = ['PairAnnihilation','ColumbScattering','BhabhaScattering','MollerScattering','ComptonScattering','PhotonCreation']

Index_List = {
    'PairAnnihilation': torch.tensor([[0,0,1,2,2,3],[1,2,2,3,4,4]],dtype=torch.int64),
    'BhabhaScattering': torch.tensor([[0,0,0,1,1,2,2,3],[1,2,3,2,4,3,4,4]],dtype=torch.int64),
    'MollerScattering': torch.tensor([[0,0,0,1,1,1,2,2],[2,3,4,2,3,4,3,4]],dtype=torch.int64),
    'ColumbScattering': torch.tensor([[0,0,1,1,2,2],[2,3,2,4,3,4]],dtype=torch.int64),
    'ComptonScattering':torch.tensor([[0,0,0,1,1,2,2,3],[1,2,3,2,3,3,4,4]],dtype=torch.int64),
    'PhotonCreation': torch.tensor([[0,0,0,1,1,1,2,2],[2,3,4,2,3,4,3,4]],dtype=torch.int64)
}

def PairAnnihilation(Ecm,charge:int, in_particle:str, out_particle:str, ang):
    graph = FeynGraph(num_nodes=5, num_edges=6, amp=0)
    edge_feat = torch.tensor(
        [[-1,1],[1,1],[1,1],[1,1],[1,1],[-1,1]],dtype=torch.float
    )
    E_in, E_out = Ecm/2, Ecm/2
    p_in, p_out = (E_in**2 - mass[in_particle]**2)**0.5, (E_out**2 - mass[out_particle]**2)**0.5
    s = Ecm**2
    t = p_in**2 + p_out**2 - 2*p_in*p_out*cos(ang)
    u = -p_in**2 - p_out**2 - 2*p_in*p_out*cos(ang)
    amp = 8/s**2 * (t**2+u**2+(mass['electron']**2+mass['muon']**2)*(2*s-mass['electron']**2-mass['muon']**2))
    node_feat = torch.tensor(
        [
            [mass[in_particle], -1, 1/2, E_in, p_in, 0,-1], 
            [mass[in_particle], 1, 1/2, E_in, p_in, np.pi,-1], 
            [mass['photon'], 0, 1, Ecm, 0, 0,0],
            [mass[out_particle], -1, 1/2, E_out, p_out, ang,1],
            [mass[out_particle], 1, 1/2, E_out, p_out, np.pi+ang,1]
        ]
        ,dtype=torch.float)
    graph.set_feat_nodes(node_feat)
    graph.set_feat_edges(edge_feat)
    graph.set_amp(amp)
    #graph.set_adj()
    return graph


def ColumbScattering(Ecm,charge:int, in_particle:str, out_particle:str, ang):
    graph = FeynGraph(num_nodes=5, num_edges=6, amp=0)
    edge_feat = torch.tensor(
        [[1,1],[-1,1],[1,1],[-1,1],[1,1],[-1,1]],dtype=torch.float
    )
    E_e = (Ecm**2+mass['electron']**2-mass['muon']**2)/(2*Ecm)
    E_mu = (Ecm**2-mass['electron']**2+mass['muon']**2)/(2*Ecm)
    p = (E_e**2-mass['electron']**2)**0.5
    be = (1-4*mass['electron']**2/Ecm**2)**0.5
    bmu = (1-4*mass['muon']**2/Ecm**2)**0.5
    amp = 1+be**2*bmu**2*cos(ang)**2+4*(mass['electron']**2+mass['muon']**2)/(Ecm**2)
    node_feat = torch.tensor(
            [
                [mass['electron'], charge, 1/2, E_e, p, 0,-1], 
                [mass['muon'], charge, 1/2, E_mu, p, np.pi,-1], 
                [mass['photon'], 0, 1, Ecm, 0, 0,0],
                [mass['electron'], charge, 1/2, E_e, p, ang,1],
                [mass['muon'], charge, 1/2, E_mu, p, np.pi+ang,1]
            ]
        ,dtype=torch.float)
    graph.set_feat_nodes(node_feat)
    graph.set_feat_edges(edge_feat)
    graph.set_amp(amp)
    #graph.set_adj()
    return graph

def BhabhaScattering(Ecm,charge:int, in_particle:str, out_particle:str, ang):
    graph = FeynGraph(num_nodes=5, num_edges=8, amp=0)
    edge_feat = torch.tensor(
        [[-1,1],[1,2],[-1,1],[1,2],[-1,1],[1,2],[1,2],[-1,1]],dtype=torch.float
    )
    E_in, E_out = Ecm/2, Ecm/2
    p_in = (E_in**2 - mass[in_particle]**2)**0.5
    p_out = p_in
    s = Ecm**2
    t=-2*p_in**2*(1-cos(ang))
    u=-2*p_in**2*(1+cos(ang))
    amp = 2*((s**2+u**2)/(t**2)+2*(u**2)/(s*t)+(u**2+t**2)/(s**2))
    node_feat = torch.tensor(
            [
                [mass[in_particle], -1, 1/2, E_in, p_in, 0,-1], 
                [mass[in_particle], 1, 1/2, E_in, p_in, np.pi,-1], 
                [mass['photon'], 0, 1, Ecm, 0, 0,0],
                [mass[in_particle], -1, 1/2, E_out, p_out, ang,1],
                [mass[in_particle], 1, 1/2, E_out, p_out, np.pi+ang,1]
            ]
        ,dtype=torch.float)
    graph.set_feat_nodes(node_feat)
    graph.set_feat_edges(edge_feat)
    graph.set_amp(amp)
    #graph.set_adj()
    return graph

def MollerScattering(Ecm,charge:int, in_particle:str, out_particle:str, ang):
    graph = FeynGraph(num_nodes=5, num_edges=8, amp=0)
    edge_feat = torch.tensor(
        [[1,2],[-1,1],[-1,1],[1,2],[-1,1],[-1,1],[1,2],[1,2]],dtype=torch.float
    )
    E_in, E_out = Ecm/2, Ecm/2
    p_in = (E_in**2 - mass[in_particle]**2)**0.5
    p_out = p_in
    s = Ecm**2
    t = -2*p_in**2*(1-cos(ang))
    u = -2*p_in**2*(1+cos(ang))
    amp = 2/(t*u)*(s**2-8*mass[in_particle]**2*s+12*mass[in_particle]**4)
    node_feat = torch.tensor(
            [
                [mass[in_particle], charge, 1/2, E_in, p_in, 0,-1], 
                [mass[in_particle], charge, 1/2, E_in, p_in, np.pi,-1], 
                [mass['photon'], 0, 1, Ecm, 0, 0,0],
                [mass[in_particle], charge, 1/2, E_out, p_out, ang,1],
                [mass[in_particle], charge, 1/2, E_out, p_out, np.pi+ang,1]
            ]
        ,dtype=torch.float)
    graph.set_feat_nodes(node_feat)
    graph.set_feat_edges(edge_feat)
    graph.set_amp(amp)
    #graph.set_adj()
    return graph

def ComptonScattering(Ecm,charge:int, in_particle:str, out_particle:str, ang):
    graph = FeynGraph(num_nodes=5, num_edges=8, amp=0)
    edge_feat = torch.tensor(
        [[1,1],[-1,2],[1,1],[1,2],[1,1],[-1,2],[1,2],[1,1]],dtype=torch.float
    )
    E = (Ecm**2 + mass[in_particle]**2) / (2*Ecm)
    p = (Ecm**2 - mass[in_particle]**2) / (2*Ecm)
    a = Ecm**2 - mass[in_particle]**2
    b = (Ecm**4 - mass[in_particle]**4) / (4*Ecm**2) + a**2 * cos(ang) / (4*Ecm**2)
    amp = 2*(a/b+b/a+2*mass[in_particle]**2*(1/a-1/b)+mass[in_particle]**4*(1/a-1/b)**2)
    node_feat = torch.tensor(
            [
                [mass[in_particle], charge, 1/2, E, p, 0,-1], 
                [mass['photon'],0, 1, p, p, np.pi,-1], 
                [mass[in_particle], charge, 1/2, Ecm, 0, 0,0],
                [mass[in_particle], charge, 1/2, E, p, ang,1],
                [0, 0, 1, p, p, np.pi+ang,1]
            ]
        ,dtype=torch.float)
    graph.set_feat_nodes(node_feat)
    graph.set_feat_edges(edge_feat)
    graph.set_amp(amp)
    return graph

def PhotonCreation(Ecm,charge:int, in_particle:str, out_particle:str, ang):
    graph = FeynGraph(num_nodes=5, num_edges=8, amp=0)
    edge_feat = torch.tensor(
        [[-1,2],[1,1],[1,1],[-1,2],[1,1],[1,1],[1,2],[1,2]],dtype=torch.float
    )
    E=Ecm/2
    p = (E**2 - mass[in_particle]**2) ** 0.5
    a = E**2 - E*p*cos(ang)
    b = E**2 + E*p*cos(ang)
    amp = 2*(a/b+b/a+2*mass[in_particle]**2*(1/a+1/b)-mass[in_particle]**4*(1/a+1/b)**2)
    node_feat = torch.tensor(
            [
                [mass[in_particle], charge, 1/2, E, p, 0,-1], 
                [mass[in_particle],-charge, 1/2, E, p, np.pi,-1], 
                [mass[in_particle], charge, 1/2, Ecm, 0, 0,0],
                [0, 0, 1, E, E, ang,1],
                [0, 0, 1, E, E, np.pi+ang,1]
            ]
        ,dtype=torch.float)
    graph.set_feat_nodes(node_feat)
    graph.set_feat_edges(edge_feat)
    graph.set_amp(amp)
    return graph

