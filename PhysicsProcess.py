import torch
import numpy as np
from FeynGraph import FeynGraph
from math import cos

mass = {'electron': 0.51099895, 'muon':105.6583775, 'photon':0}

Proc_List = ['PairAnnihilation','ColumbScattering','BhabhaScattering','MollerScattering']

Index_List = {
    'PairAnnihilation': torch.tensor([[0,0,1,2,2,3],[1,2,2,3,4,4]],dtype=torch.int64),
    'BhabhaScattering': torch.tensor([[0,0,0,1,1,2,2,3],[1,2,3,2,4,3,4,4]],dtype=torch.int64),
    'MollerScattering': torch.tensor([[0,0,0,1,1,1,2,2],[2,3,4,2,3,4,3,4]],dtype=torch.int64),
    'ColumbScattering': torch.tensor([[0,0,1,1,2,2],[2,3,2,4,3,4]],dtype=torch.int64)
}

def PairAnnihilation(Ecm,charge:int, in_particle:str, out_particle:str, out_ang):
    if Ecm>mass[in_particle]:
        if Ecm>mass[out_particle]:
            pass 
        else:
            raise ValueError('PairAnnihilation: energy in center-of-mass frame shall be greater than the mass of incoming particle. The mass of incoming particle is '+str(mass[in_particle])+', and the energy is '+str(Ecm)+'.')
    else:
        raise ValueError('PairAnnihilation: energy in center-of-mass frame shall be greater than the mass of incoming particle. The mass of incoming particle is '+str(mass[in_particle])+', and the energy is '+str(Ecm)+'.')
    graph = FeynGraph(num_nodes=5, num_edges=6, amp=0)
    adj = torch.tensor([[0,0,1,2,2,3],[1,2,2,3,4,4]],dtype=torch.int64)
    edge_feat = torch.tensor(
        [[-1,1],[1,1],[1,1],[1,1],[1,1],[-1,1]],dtype=torch.float
    )
    E_in, E_out = Ecm/2, Ecm/2
    p_in, p_out = (E_in**2 - mass[in_particle]**2)**0.5, (E_out**2 - mass[out_particle]**2)**0.5
    s = Ecm**2
    t = p_in**2 + p_out**2 - 2*p_in*p_out*cos(out_ang)
    u = -p_in**2 - p_out**2 - 2*p_in*p_out*cos(out_ang)
    amp = 8/s**2 * (t**2+u**2+(mass['electron']**2+mass['muon']**2)*(2*s-mass['electron']**2-mass['muon']**2))
    node_feat = torch.tensor(
        [
            [mass[in_particle], -1, 1/2, E_in, p_in, 0], 
            [mass[in_particle], 1, 1/2, E_in, p_in, np.pi], 
            [mass['photon'], 0, 1, Ecm, 0, 0],
            [mass[out_particle], -1, 1/2, E_out, p_out, out_ang],
            [mass[out_particle], 1, 1/2, E_out, p_out, np.pi+out_ang]
        ]
        ,dtype=torch.float)
    graph.set_feat_nodes(node_feat)
    graph.set_feat_edges(edge_feat)
    graph.set_amp(amp)
    #graph.set_adj()
    return graph


def ColumbScattering(Ecm,charge:int, in_particle:str, out_particle:str, out_ang):
    if Ecm**2>mass['muon']**2-mass['electron']**2:
        pass
    else:
        raise ValueError('ColumbScattering: energy in center-of-mass frame shall be greater than '+str((mass['muon']**2-mass['electron']**2)**0.5)+', but got '+str(Ecm)+'.')
    graph = FeynGraph(num_nodes=5, num_edges=6, amp=0)
    torch.tensor([[0,0,1,1,2,2],[2,3,2,4,3,4]],dtype=torch.int64)
    edge_feat = torch.tensor(
        [[1,1],[-1,1],[1,1],[-1,1],[1,1],[-1,1]],dtype=torch.float
    )
    E_e = (Ecm**2+mass['electron']**2-mass['muon']**2)/(2*Ecm)
    E_mu = (Ecm**2-mass['electron']**2+mass['muon']**2)/(2*Ecm)
    p = (E_e**2-mass['electron']**2)**0.5
    be = (1-4*mass['electron']**2/Ecm**2)**0.5
    bmu = (1-4*mass['muon']**2/Ecm**2)**0.5
    amp = 1+be**2*bmu**2*cos(out_ang)**2+4*(mass['electron']**2+mass['muon']**2)/(Ecm**2)
    node_feat = torch.tensor(
            [
                [mass['electron'], charge, 1/2, E_e, p, 0], 
                [mass['muon'], charge, 1/2, E_mu, p, np.pi], 
                [mass['photon'], 0, 1, Ecm, 0, 0],
                [mass['electron'], charge, 1/2, E_e, p, out_ang],
                [mass['electron'], charge, 1/2, E_mu, p, np.pi+out_ang]
            ]
        ,dtype=torch.float)
    graph.set_feat_nodes(node_feat)
    graph.set_feat_edges(edge_feat)
    graph.set_amp(amp)
    #graph.set_adj()
    return graph

def BhabhaScattering(Ecm,charge:int, in_particle:str, out_particle:str, out_ang):
    if Ecm>mass[in_particle]:
        pass
    else:
        raise ValueError('BhabhaScattering: energy in center-of-mass frame shall be greater than mass of the incoming particle ('+str((mass['muon']**2-mass['electron']**2)**0.5)+'), but got '+str(Ecm)+'.')
    graph = FeynGraph(num_nodes=5, num_edges=8, amp=0)
    adj = torch.tensor([[0,0,0,1,1,2,2,3],[1,2,3,2,4,3,4,4]],dtype=torch.int64)
    edge_feat = torch.tensor(
        [[-1,1],[1,2],[-1,1],[1,2],[-1,1],[1,2],[1,2],[-1,1]],dtype=torch.float
    )
    E_in, E_out = Ecm/2, Ecm/2
    p_in = (E_in**2 - mass[in_particle]**2)**0.5
    p_out = p_in
    s = Ecm**2
    t=-2*p_in**2*(1-cos(out_ang))
    u=-2*p_in**2*(1+cos(out_ang))
    amp = 2*((s**2+u**2)/(t**2)+2*(u**2)/(s*t)+(u**2+t**2)/(s**2))
    node_feat = torch.tensor(
            [
                [mass[in_particle], -1, 1/2, E_in, p_in, 0], 
                [mass[in_particle], 1, 1/2, E_in, p_in, np.pi], 
                [mass['photon'], 0, 1, Ecm, 0, 0],
                [mass[in_particle], -1, 1/2, E_out, p_out, out_ang],
                [mass[in_particle], 1, 1/2, E_out, p_out, np.pi+out_ang]
            ]
        ,dtype=torch.float)
    graph.set_feat_nodes(node_feat)
    graph.set_feat_edges(edge_feat)
    graph.set_amp(amp)
    #graph.set_adj()
    return graph

def MollerScattering(Ecm,charge:int, in_particle:str, out_particle:str, out_ang):
    if Ecm>mass[in_particle]:
        pass
    else:
        raise ValueError('MøllerScattering: energy in center-of-mass frame shall be greater than mass of the incoming particle ('+str((mass['muon']**2-mass['electron']**2)**0.5)+'), but got '+str(Ecm)+'.')
    graph = FeynGraph(num_nodes=5, num_edges=8, amp=0)
    adj = torch.tensor([[0,0,0,1,1,1,2,2],[2,3,4,2,3,4,3,4]],dtype=torch.int64)
    edge_feat = torch.tensor(
        [[1,2],[-1,1],[-1,1],[1,2],[-1,1],[-1,1],[1,2],[1,2]],dtype=torch.float
    )
    E_in, E_out = Ecm/2, Ecm/2
    p_in = (E_in**2 - mass[in_particle]**2)**0.5
    p_out = p_in
    s = Ecm**2
    t = -2*p_in**2*(1-cos(out_ang))
    u = -2*p_in**2*(1+cos(out_ang))
    amp = 2/(t*u)*(s**2-8*mass[in_particle]**2*s+12*mass[in_particle]**4)
    node_feat = torch.tensor(
            [
                [mass[in_particle], charge, 1/2, E_in, p_in, 0], 
                [mass[in_particle], charge, 1/2, E_in, p_in, np.pi], 
                [mass['photon'], 0, 1, Ecm, 0, 0],
                [mass[in_particle], charge, 1/2, E_out, p_out, out_ang],
                [mass[in_particle], charge, 1/2, E_out, p_out, np.pi+out_ang]
            ]
        ,dtype=torch.float)
    graph.set_feat_nodes(node_feat)
    graph.set_feat_edges(edge_feat)
    graph.set_amp(amp)
    #graph.set_adj()
    return graph


