import torch
import numpy as np
from FeynGraph import FeynGraph
from math import cos
from scipy.optimize import fsolve

MASS = {'electron': 0.51099895, 'muon':105.6583775, 'photon':0}

PROC_LIST = ['PairAnnihilation', 'BhabhaScattering','MollerScattering','CoulombScattering','ComptonScattering']

ADJ_LIST = {
    "s_channel": np.array([[1,1,1,0,0],[1,1,1,0,0],[1,1,1,1,1],[0,0,1,1,1],[0,0,1,1,1]]),
    "u_channel": np.array([[1,0,1,1,0],[0,1,1,0,1],[1,1,1,1,1],[1,0,1,1,0],[0,1,1,0,1]]),
    "t_channel": np.array([[1,0,1,0,1],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[1,0,1,0,1]]),
    #'PairAnnihilation': torch.tensor([[0,0,1,2,2,3],[1,2,2,3,4,4]],dtype=torch.int64),
    'PairAnnihilation': torch.tensor([[1,1,1,0,0],[1,1,1,0,0],[1,1,1,1,1],[0,0,1,1,1],[0,0,1,1,1]],dtype=torch.int64),
    #'BhabhaScattering': torch.tensor([[0,0,0,1,1,2,2,3],[1,2,3,2,4,3,4,4]],dtype=torch.int64),
    'BhabhaScattering': torch.tensor([[1,1,1,1,0],[1,1,1,0,1],[1,1,1,1,1],[1,0,1,1,1],[0,1,1,1,1]],dtype=torch.int64),
    #'MollerScattering': torch.tensor([[0,0,0,1,1,1,2,2],[2,3,4,2,3,4,3,4]],dtype=torch.int64),
    'MollerScattering': torch.tensor([[1,0,1,1,1],[0,1,1,1,1],[1,1,1,1,1],[1,1,1,1,0],[1,1,1,0,1]],dtype=torch.int64),
    #'CoulombScattering': torch.tensor([[0,0,1,1,2,2],[2,3,2,4,3,4]],dtype=torch.int64),
    'CoulombScattering': torch.tensor([[1,0,1,1,0],[0,1,1,0,1],[1,1,1,1,1],[1,0,1,1,0],[0,1,1,0,1]],dtype=torch.int64),
    #'ComptonScattering':torch.tensor([[0,0,0,1,1,2,2,3],[1,2,3,2,3,3,4,4]],dtype=torch.int64),
    'ComptonScattering':torch.tensor([[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,1],[0,1,1,1,1],[0,1,1,1,1]],dtype=torch.int64),
    #'PhotonCreation': torch.tensor([[0,0,0,1,1,1,2,2],[2,3,4,2,3,4,3,4]],dtype=torch.int64)
    'PhotonCreation': torch.tensor([[1,0,1,1,1],[0,1,1,1,1],[1,1,1,1,1],[1,1,1,1,0],[1,1,1,0,1]],dtype=torch.int64)
}

def rand_mom4(num_in, num_out, mass_in, mass_out, sig_mom=1000, seed=None):
    """ 
    return the 4-momentum for n external legs as a list of numpy array. mass_list should be np.ndarray.
    sig_mom are parameters of the normal distribution of momentum.
    """
    np.random.seed(seed)
    if num_in+num_out > 2:
        pass 
    else:
        raise ValueError('n must bigger than 2')
    # generate momentum for incoming particle
    p_in = np.reshape(np.random.normal(0, sig_mom, 3*(num_in-1)), (num_in-1, 3))
    p_in = np.append(p_in, -np.sum(p_in, axis=0).reshape((1, 3)), axis=0)
    E_in = np.array([
        np.sqrt(np.linalg.norm(p_in[i])**2 + mass_in[i]**2) for i in range(num_in)
    ])
    mom_in = np.array(
        [[E_in[i], p_in[i, 0], p_in[i, 1], p_in[i, 2]] for i in range(num_in)]
    )
    # generate momentum for outgoing particle
    key = 1
    while key == 1:
        p_out = np.reshape(np.random.normal(0, sig_mom/2, 3*(num_out-1)), (num_out-1, 3))
        p_out = np.append(p_out, -np.sum(p_out, axis=0).reshape((1,3)), axis=0)
        E_out = np.array([
            np.sqrt(np.linalg.norm(p_out[i])**2 + mass_out[i]**2) for i in range(num_out)
        ])
        if ((mass_out[-1]**2 + p_out[-1, 1]**2 + p_out[-1, 2]**2)**0.5 + 
            (mass_out[num_out-2]**2 + p_out[num_out-2, 1]**2 + p_out[num_out-2, 2]**2)**0.5 +
            np.sum(E_out[0: num_out-2]) - np.sum(E_in)) < 0:
            key = 0

    def f(x):
        x1, x2 = x 
        eq1 = (
            (x1**2 + mass_out[-1]**2 + p_out[-1, 1]**2 + p_out[-1, 2]**2)**0.5 + 
            (x2**2 + mass_out[num_out-2]**2 + p_out[num_out-2, 1]**2 + p_out[num_out-2, 2]**2)**0.5 - 
            np.sum(E_in) + np.sum(E_out[0: num_out-2])
        )
        eq2 = x1 + x2 + np.sum(p_out[0:num_out-2, 0]) 
        return [eq1, eq2]
    #p0 = [p_out[num_out-1, 0], p_out[num_out-2, 0]]
    p0 = [0, 0]
    p_result = fsolve(f, p0) 
    p_out[num_out-1, 0], p_out[num_out-2, 0] = p_result[0], p_result[1]
    E_out = np.array([
        np.sqrt(np.linalg.norm(p_out[i])**2 + mass_out[i]**2) for i in range(num_out)
    ])
    mom_out = np.array(
        [[E_out[i], p_out[i, 0], p_out[i, 1], p_out[i, 2]] for i in range(num_out)]
    )
    return mom_in, mom_out

def minkowski_dot(p1, p2):
    return p1[0]*p2[0]-p1[1]*p2[1]-p1[2]*p2[2]-p1[3]*p2[3]

def PairAnnihilation(particle, charge, seed=None):
    p_in, p_out = rand_mom4(2,2, [MASS['muon'], MASS['muon']], [MASS['electron'], MASS['electron']], seed=seed)
    s = minkowski_dot(p_in[0]+p_in[1], p_in[0]+p_in[1])
    amp = 8/s**2 * (
        minkowski_dot(p_in[0], p_out[0]) * minkowski_dot(p_in[1], p_out[1]) + 
        minkowski_dot(p_in[0], p_out[1]) * minkowski_dot(p_in[1], p_out[0]) + 
        MASS['electron']**2 * minkowski_dot(p_out[0], p_out[1]) + 
        MASS['muon']**2 * minkowski_dot(p_in[0], p_in[1]) +
        2 * MASS['electron']**2 * MASS['muon']**2
    )
    adj = np.array([[1,1,1,0,0],[1,1,1,0,0],[1,1,1,1,1],[0,0,1,1,1],[0,0,1,1,1]])
    temp0 = [MASS['muon'], -1, 1/2]
    temp1 = [MASS['muon'], 1, 1/2]
    temp2 = [MASS['electron'], -1, 1/2]
    temp3 = [MASS['electron'], 1, 1/2]
    node_feat = np.array(
        [
            temp0 + list(p_in[0]),
            temp1 + list(p_in[1]),
            [MASS['photon'], 0, 1] + list(p_in[0]+p_in[1]),
            temp2 + list(p_out[0]),
            temp3 + list(p_out[1])
        ]
    )
    return adj, node_feat, amp

def CoulombScattering(particle, charge, seed=None):
    p_in, p_out = rand_mom4(2,2, [MASS['electron'], MASS['muon']], [MASS['electron'], MASS['muon']], seed=seed)
    amp = 8/(minkowski_dot(p_in[0]-p_out[0], p_in[0]-p_out[0])**2) * (
        minkowski_dot(p_in[0], p_in[1]) * minkowski_dot(p_out[0], p_out[1]) +
        minkowski_dot(p_in[0], p_out[1]) * minkowski_dot(p_in[1], p_out[0]) -
        MASS['electron']**2 * minkowski_dot(p_in[1], p_out[1]) -
        MASS['muon']**2 * minkowski_dot(p_in[0], p_out[1]) +
        2 * MASS['electron']**2 * MASS['muon']**2
    )
    adj = np.array([[1,0,1,1,0],[0,1,1,0,1],[1,1,1,1,1],[1,0,1,1,0],[0,1,1,0,1]])
    temp0 = [MASS['electron'], charge, 1/2]
    temp1 = [MASS['muon'], -charge, 1/2]
    temp2 = [MASS['electron'], charge, 1/2]
    temp3 = [MASS['muon'], -charge, 1/2]
    node_feat = np.array(
        [
            temp0 + list(p_in[0]),
            temp1 + list(p_in[1]),
            [MASS['photon'], 0, 1] + list(p_in[0]+p_in[1]),
            temp2 + list(p_out[0]),
            temp3 + list(p_out[1])
        ]
    )
    return adj, node_feat, amp

def BhabhaScattering(particle, charge, seed=None):
    p_in, p_out = rand_mom4(2,2, [MASS[particle], MASS[particle]], [MASS[particle], MASS[particle]], seed=seed)
    s = minkowski_dot(p_in[0]+p_in[1],p_in[0]+p_in[1])
    t = minkowski_dot(p_in[0]-p_out[0], p_in[0]-p_out[0])
    u = minkowski_dot(p_in[0]-p_out[1], p_in[0]-p_out[1])
    amp = 2*(t**2+u**2)/s**2
    adj = np.array([
        [1,1,1,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0,0],
        [1,1,1,1,1,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,1,0,1,1,0],
        [0,0,0,0,0,0,1,1,0,1],
        [0,0,0,0,0,1,1,1,1,1],
        [0,0,0,0,0,1,0,1,1,0],
        [0,0,0,0,0,0,1,1,0,1]
    ])
    temp0 = [MASS[particle], 1, 1/2]
    temp1 = [MASS[particle], -1, 1/2]
    nodes_feat = np.array([
        temp0 + list(p_in[0]),
        temp1 + list(p_in[1]),
        [0, 0, 1] + list(p_in[0]+p_in[1]),
        temp0 + list(p_out[0]),
        temp1 + list(p_out[1]),
        temp0 + list(p_in[0]),
        temp1 + list(p_in[1]),
        [0, 0, 1] + list(p_in[0]+p_in[1]),
        temp0 + list(p_out[0]),
        temp1 + list(p_out[1])
    ])
    return adj, nodes_feat, amp

def MollerScattering(particle, charge, seed=None):
    p_in, p_out = rand_mom4(2,2, [MASS[particle], MASS[particle]], [MASS[particle], MASS[particle]], seed=seed)
    s = minkowski_dot(p_in[0]+p_in[1],p_in[0]+p_in[1])
    t = minkowski_dot(p_in[0]-p_out[0], p_in[0]-p_out[0])
    u = minkowski_dot(p_in[0]-p_out[1], p_in[0]-p_out[1])
    amp = (2/(t*u))*(s**2-8*MASS[particle]**2*s+12*MASS[particle]**4)
    #"u_channel": np.array([[1,0,1,1,0],[0,1,1,0,1],[1,1,1,1,1],[1,0,1,1,0],[0,1,1,0,1]]),
    #"t_channel": np.array([[1,0,1,0,1],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[1,0,1,0,1]]),
    adj = np.array([
        [1,0,1,1,0,0,0,0,0,0],
        [0,1,1,0,1,0,0,0,0,0],
        [1,1,1,1,1,0,0,0,0,0],
        [1,0,1,1,0,0,0,0,0,0],
        [0,1,1,0,1,0,0,0,0,0],
        [0,0,0,0,0,1,0,1,0,1],
        [0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,1,1,1,1,1],
        [0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,1,0,1,0,1]
    ])
    temp = [MASS[particle], charge, 1/2]
    nodes_feat = np.array([
        temp + list(p_in[0]),
        temp + list(p_in[1]),
        [0,0,1] + list(p_in[0]+p_in[1]),
        temp + list(p_out[0]),
        temp + list(p_out[1]),
        temp + list(p_in[0]),
        temp + list(p_in[1]),
        [0,0,1] + list(p_in[0]+p_in[1]),
        temp + list(p_out[0]),
        temp + list(p_out[1]),
    ])
    return adj, nodes_feat, amp

def ComptonScattering(particle, charge, seed=None):
    #"t_channel": np.array([[1,0,1,1,0],[0,1,1,0,1],[1,1,1,1,1],[1,0,1,1,0],[0,1,1,0,1]]),
    #"u_channel": np.array([[1,0,1,0,1],[0,1,1,1,0],[1,1,1,1,1],[0,1,1,1,0],[1,0,1,0,1]]),
    p_in, p_out = rand_mom4(2, 2, [MASS[particle], 0], [MASS[particle], 0], seed=seed)
    amp = 2 * (
        minkowski_dot(p_in[0], p_in[1]) / minkowski_dot(p_in[0], p_out[1]) + 
        minkowski_dot(p_in[0], p_out[1]) / minkowski_dot(p_in[0], p_in[1]) + 
        2*MASS[particle]**2 * (1/minkowski_dot(p_in[0], p_in[1]) - 1/minkowski_dot(p_in[0], p_out[1])) +
        MASS[particle]**4 * (1/minkowski_dot(p_in[0], p_in[1]) - 1/minkowski_dot(p_in[0], p_out[1]))**2
    )
    adj = np.array([
        [1,1,1,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0,0],
        [1,1,1,1,1,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,0,0],
        [0,0,0,0,0,1,0,1,0,1],
        [0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,1,1,1,1,1],
        [0,0,0,0,0,0,1,1,1,0],
        [0,0,0,0,0,1,0,1,0,1]
    ])
    temp0 = [MASS[particle], charge, 1/2]
    temp1 = [0, 0, 1]
    nodes_feat = np.array(
        [
            temp0 + list(p_in[0]),
            temp1 + list(p_in[1]),
            [MASS[particle], charge, 1/2] + list(p_in[0]+p_in[1]),
            temp0 + list(p_out[0]),
            temp1 + list(p_out[1]),
            temp0 + list(p_in[0]),
            temp1 + list(p_in[1]),
            [MASS[particle], charge, 1/2] + list(p_in[0]+p_in[1]),
            temp1 + list(p_out[0]),
            temp0 + list(p_out[1]),
        ]
    )
    return adj, nodes_feat, amp

def PhotonCreation(particle, charge, seed=None):
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

if __name__ == '__main__':
    print(PairAnnihilation(1,1,seed=1)[2])
