from MIP.kp import generateUniformKPND, generateWeaklyCorrelatedKPND, generateStronglyCorrelatedKPND, kp_greedy
from MIP.kp import rndGenerateUniformKPND
from MIP.problem import random_coalitions, Problem
from lexcell import lex_cell, adv_lex_cell
from tqdm import tqdm

import sys,os,math,subprocess
import torch
#import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
#import torch_optimizer as optim
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

from operator import itemgetter




def generate_mats (n,l,nd,ncoal):
    individuals = set(range(n))
    kp = rndGenerateUniformKPND(n,1000,nd)
    mat = kp.toAdjacencyMatrix()
    coals = random_coalitions(individuals, l,ncoal)
    scores,sols = zip(*(list (map (kp.solve_coalition, coals))))

    order = adv_lex_cell(individuals, list(zip(sols,coals)), scores)

    opt,real_sol = kp.solve()

    adv_lex_greed, lex_sol = kp.greedy(order)

    real_order = kp.trivial_order()
    real_greed, _ =kp.greedy(real_order)


    return kp, mat, lex_sol, torch.tensor(real_sol), adv_lex_greed, real_greed, opt
 
def generate_validset(n, l, m, ncoal):
    cs = []
    As = []
    bs = []
    mats = []

    tgts_sols = []
    tgts_opts = []

    tgts_sols_real = []
    tgts_opts_real = []


    for i in tqdm(range(100)):
        kp, mat, lex_sol, real_sol, lex_opt, _, real_opt = generate_mats(n,l,m,ncoal)

        c, A, b = kp.toTensor()
        cs.append(c)
        As.append(A)
        bs.append(b)

        mats.append(mat)
        tgts_sols_real.append(real_sol)
        tgts_opts_real.append(real_opt)
        tgts_sols.append(lex_sol)
        tgts_opts.append(lex_opt)
    torch.save(cs, f"valid/valid-cs-{n}-{l}-{m}-{ncoal}.pt")
    torch.save(As, f"valid/valid-As-{n}-{l}-{m}-{ncoal}.pt")
    torch.save(bs, f"valid/valid-bs-{n}-{l}-{m}-{ncoal}.pt")
    torch.save(mats, f"valid/valid-mats-{n}-{l}-{m}-{ncoal}.pt")
    torch.save(tgts_sols_real, f"valid/valid-sols-{n}-{l}-{m}-{ncoal}-True.pt")
    torch.save(tgts_opts_real, f"valid/valid-opts-{n}-{l}-{m}-{ncoal}-True.pt")
    torch.save(tgts_sols, f"valid/valid-sols-{n}-{l}-{m}-{ncoal}-False.pt")
    torch.save(tgts_opts, f"valid/valid-opts-{n}-{l}-{m}-{ncoal}-False.pt")

    #return kps, torch.stack(mats), tgts_sols, tgts_opts    


class KPValidSet(Dataset):
    def __init__(self, cs, As, bs, mats, sols, tgts):
        self.mats = mats
        self.sols = sols
        self.opts = tgts

        self.cs = cs
        self.As = As
        self.bs = bs

    def __len__(self):
        return self.mats.size(0)
    def __getitem__(self, idx):
        pb = Problem(list(self.cs[idx]), list(self.As[idx]), list (self.bs[idx]))
        return pb, self.mats[idx], self.sols[idx], self.opts[idx]

def load_validset(n,l,m,ncoal,real):
    cs = torch.load(f"valid/valid-cs-{n}-{l}-{m}-{ncoal}.pt")
    As=torch.load(f"valid/valid-As-{n}-{l}-{m}-{ncoal}.pt")
    bs=torch.load(f"valid/valid-bs-{n}-{l}-{m}-{ncoal}.pt")
    mats=torch.load(f"valid/valid-mats-{n}-{l}-{m}-{ncoal}.pt")
    tgts_sols=torch.load(f"valid/valid-sols-{n}-{l}-{m}-{ncoal}-{real}.pt")
    tgts_opts=torch.load(f"valid/valid-opts-{n}-{l}-{m}-{ncoal}-{real}.pt")
    return KPValidSet(cs,As,bs,mats,tgts_sols,tgts_opts)

    
if __name__=='__main__':
    generate_validset(50,25,10,100)

    print (load_validset(50,25,10,100,False))
    print (load_validset(50,25,10,100,True))

