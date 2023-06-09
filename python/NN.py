#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from MIP.kp import generateUniformKPND, generateWeaklyCorrelatedKPND, generateStronglyCorrelatedKPND, kp_greedy
from MIP.kp import rndGenerateUniformKPND
from MIP.problem import random_coalitions, Problem
from lexcell import lex_cell, adv_lex_cell
from tqdm import tqdm

from validset import load_validset

import sys,os,math,subprocess
import torch
#import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
#import torch_optimizer as optim
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from operator import itemgetter

from NN.blocks import ResidualLinear, GraphConvolution, GCNLinear, GCNResnetLinear, GCNResidual
from NN.dataset import KPDataset

n=50
m=10
hid=2000
nbhid=20
l = 25
ncoal=100
real = True
plot_hid_grad = False
save = True

datalen=300000
testlen=100


gpu="cuda:1"
lr=1e-5
#lr=1e-6
comm = f"{gpu}-gcn_nolin-ln-dropout-mlp-adamW-{nbhid}*{hid}-{datalen}"
use_cuda = torch.cuda.is_available()

#device = torch.device("cuda:1" if use_cuda else "cpu")
device = None
if use_cuda:
    #device = torch.device(auto_gpu_selection())
    device = torch.device(gpu)
else:
    device = torch.device("cpu")

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
 
def generate_validset():
    kps = []
    mats = []
    tgts_sols = []
    tgts_opts = []


    for i in tqdm(range(100)):
    #for i in tqdm(range(testlen)):
        kp, mat, lex_sol, real_sol, lex_opt, _, real_opt = generate_mats(n,l,m,ncoal)
        kps.append(kp)
        mats.append(mat)
        if real:
            tgts_sols.append(real_sol)
            tgts_opts.append(real_opt)
        else:
            tgts_sols.append(lex_sol)
            tgts_opts.append(lex_opt)
    return kps, torch.stack(mats), tgts_sols, tgts_opts    

def evaluate_accuracy(model,validset):


    sol_accuracy, opt_accuracy = [],[]
    real_opt_accuracy = []

    for entry in validset:
        kp, mat, tgt_sol, tgt_opt, real_opt = entry
        pred = model(torch.stack([mat]).to(device)).cpu()

        order = map (itemgetter(1),sorted(zip(pred[0], range(n)), reverse=True))
        opt, sol = kp.greedy(order)
        sol = sol.type(torch.int64)
        opt_acc = opt/tgt_opt
        #sol_acc = tgt_sol.gather(0,sol).sum()/tgt_sol.sum()
        sol_acc = tgt_sol[sol==1].sum()/tgt_sol.sum()
        real_opt_acc = opt/real_opt
        opt_accuracy.append(opt_acc)
        sol_accuracy.append(sol_acc)
        real_opt_accuracy.append(real_opt_acc)
        
    sol_accuracy = torch.tensor(sol_accuracy)
    opt_accuracy = torch.tensor(opt_accuracy)
    real_opt_accuracy = torch.tensor(real_opt_accuracy)
    return sol_accuracy.mean(), opt_accuracy.mean(), real_opt_accuracy.mean()


class TestNet (nn.Module):
    def __init__ (self, n, m):
        super (TestNet, self).__init__()

        self.n = n
        self.m = m

        self.initial_fts = torch.cat((torch.ones(n), torch.zeros(m))).reshape(1,n+m,1).to(device)
        self.gcninit = GCNLinear(1, hid)

        mlist = []
        for i in range(nbhid):
            mlist.append(GCNResidual(hid))
        self.hid = nn.Sequential(*mlist)
        self.last = nn.Linear(hid*(n+m),n)

    def forward (self, x):
        self.train()
        bsize = x.size()[0]
        inputs = self.initial_fts.repeat(bsize,1,1)
        initx, adjs = self.gcninit((inputs,x))
        initx = F.relu(initx)
        
        x, _ = self.hid((initx,adjs))
        y = self.last(x.reshape(bsize,hid*(self.n+self.m)))
        return y

        #x,_ = self.gcnhid1((initx,adjs))
        #x,_ = self.gcnhid2((x,adjs))
        #x = self.drop1(x)
        #x, _ = self.gcnhid3((x,adjs))
        #x, _ = self.gcnhid4((x,adjs))
        #x = self.drop2(x)
        ###### return avec mlp 
        #y = self.gcnlast(x.reshape(bsize, hid*(self.n+self.m)))
        return y
        #y,_ = self.gcnlast(x) #return avec gcnlinear



        #return y[:,:n].reshape(bsize,n) #classic gcn
        #maxvals, maxids = y.max(dim=1)
        #y = self.linlast(maxvals)
        #return y
        #return y.sum(dim=1)/y.size(1)


   
def gradient_norm(model):
    total_norm = 0
    for param in model.parameters():
        if (None != param.grad):
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item()**2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_accuracy(output, target):
    ks = target.sum(dim=-1)
    total = torch.tensor(0).float().to(device)
    for i, k in enumerate(ks):
        _, top_k = output[i].topk(int(k.item()))
        selected = target[i][top_k].sum() / target[i].sum()
        total += selected
    return total / target.size(0)  
        
    #_,kbest=output.topk(, dim=-1)
    #return target.gather(-1, kbest).sum(dim=1).mean()

    
def train_gcn(model,epoch, batch_size, real=False):
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    writer = SummaryWriter(comment=f"n-100_p-10_rnd-{comm}-{real}")

    trainset=None
    if datalen == 300000:
        print ("loading large dataset")
        if real:
            trainset=torch.load("dataset_50_10_25_300000_real.pt")
        else:
            trainset=torch.load("dataset_50_10_25_300000.pt")
            
    else:
        trainset = KPDataset(n,m,l,datalen,real)
    testset = KPDataset(n,m,l,testlen,real)
    best = 10^5
   
    print ("loaded")
    trainset = DataLoader(trainset, batch_size = batch_size, shuffle=True)
    testset = DataLoader(testset, batch_size = testlen, shuffle=True)

    print ("load validation_set")
    validset = load_validset(n,l,m,ncoal,real)

    count=0
    for e in range(epoch):
        for batch in trainset:
        #for batch in validset:
            optimizer.zero_grad()
            loss = 0

            x, f_x = batch
            #_, x,f_x, _ = batch
            #x = torch.stack([x])
            #f_x = torch.stack([f_x])

            x = x.to(device)
            f_x = f_x.to(device)

            y = model.forward(x)

            loss = F.binary_cross_entropy_with_logits(y, f_x)
            loss.backward()
            optimizer.step() # un pas de la descente de gradient

            writer.add_scalar("loss/train", loss, count)
            writer.add_scalar("grad/norm", gradient_norm(model), count)
            count += 1
            if plot_hid_grad:
                writer.add_scalar(f'grad/init', gradient_norm(model.gcninit), count)
                writer.add_scalar(f'grad/hid-{1}', gradient_norm(model.gcnhid1), count)
                writer.add_scalar(f'grad/hid-{2}', gradient_norm(model.gcnhid2), count)
                writer.add_scalar(f'grad/last', gradient_norm(model.gcnlast), count)
                #for i, (name,layer)  in enumerate(model.gchid.named_children()):
                #     writer.add_scalar(f'hid/hid-{i}', gradient_norm(layer), count)

        with torch.no_grad():
            sol_acc, opt_acc, real_acc = evaluate_accuracy(model,validset)
            writer.add_scalar("accuracy/sol", sol_acc, e)
            writer.add_scalar("accuracy/opt", opt_acc, e)
            writer.add_scalar("accuracy/real", real_acc, e)
            if real_acc > best:
                    best = accuracy
                    if save:
                        torch.save(model.state_dict(), f"models/model_sol_{n}-{datalen}-{real}-{comm}.pt")


            for batch in testset:
                x, f_x = batch
                x = x.to(device)
                f_x = f_x.to(device)
                output = model.forward(x)
                loss = F.binary_cross_entropy_with_logits(output, f_x)
                #loss = lossfunction(testtarget, output)
                writer.add_scalar("loss/test", loss, e)

                accuracy = compute_accuracy(output, f_x)

                #output = torch.sigmoid(output)
                #output[output>0.5]=1
                #output[output<=0.5]=0
                #print (output[0])
                #accuracy = (n - (output-f_x).abs().sum(dim=1)).mean()
                writer.add_scalar("loss/accuracy", accuracy , e)

                #if accuracy > best:
                #    best = accuracy
                #    if save:
                #        torch.save(model.state_dict(), f"model_sol_{n}-{datalen}-{real}-{comm}.pt")



def show_params(model):
    for param in model.parameters():
        print (param)

        
if __name__ == '__main__':
    if not sys.argv[2:]:
        sys.stdout.write("Size needed in argument : epoch batchsize\n")
        sys.exit(0)

    torch.set_default_dtype(torch.float32)

    batchsize = int(sys.argv[2])
 
    
    print("Training on " + str(device))
    print("before training")
    model = TestNet(n,m) # predicts the solution
    model.to(device)
    train_gcn(model,int(sys.argv[1]),int(sys.argv[2]), real)
    #model = TestNet(objs/1000,A/1000,b/(n*1000), 1).to(device) # predicts the optimal value
    #train_opt(model,int(sys.argv[1]),int(sys.argv[2]),F.mse_loss)
    #show_params(model)
    #train(5,int(1e7))
    print("after training")
    show_params(model)


