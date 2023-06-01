#! /usr/bin/env python3
# -*- coding: utf-8 -*-

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

from NN.blocks import ResidualLinear, GraphConvolution, GCNLinear, GCNResnetLinear, GCNResidual
from NN.dataset import KPDataset

n=50
m=10
hid=10000
l = 25
real = False
plot_hid_grad = False
save = True

datalen=100000
testlen=100


gpu="cuda:0"
lr=1e-5
#lr=1e-6
comm = f"{gpu}-gcn_nolin-dropout-mlp-adamW"
use_cuda = torch.cuda.is_available()

#device = torch.device("cuda:1" if use_cuda else "cpu")
device = None
if use_cuda:
    #device = torch.device(auto_gpu_selection())
    device = torch.device(gpu)
else:
    device = torch.device("cpu")



class TestNet (nn.Module):
    def __init__ (self, n, m):
        super (TestNet, self).__init__()

        self.n = n
        self.m = m

        self.initial_fts = torch.cat((torch.ones(n), torch.zeros(m))).reshape(1,n+m,1).to(device)
        self.gcninit = GCNLinear(1, hid)
        #self.gcnhid1 = GCNResnetLinear(hid)
        #self.gcnhid2 = GCNResnetLinear(hid)
        self.gcnhid1 = GCNResidual(hid)
        self.gcnhid2 = GCNResidual(hid)
        self.drop1 = nn.Dropout()
        self.gcnhid3 = GCNResidual(hid)
        self.gcnhid4 = GCNResidual(hid)
        self.drop2 = nn.Dropout()
        self.gcnlast = nn.Linear(hid*(n+m), n)
        #self.gcnlast = GCNLinear(hid,n)
        #self.linlast = nn.Linear(n,n)


    def forward (self, x):
        self.train()
        bsize = x.size()[0]
        inputs = self.initial_fts.repeat(bsize,1,1)
        initx, adjs = self.gcninit((inputs,x))
        initx = F.relu(initx)
        x,_ = self.gcnhid1((initx,adjs))
        x,_ = self.gcnhid2((x,adjs))
        x = self.drop1(x)
        x, _ = self.gcnhid3((x,adjs))
        x, _ = self.gcnhid4((x,adjs))
        x = self.drop2(x)
        ##### return avec mlp 
        y = self.gcnlast(x.reshape(bsize, hid*(self.n+self.m)))
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
    trainset = KPDataset(n,m,l,datalen,real)
    testset = KPDataset(n,m,l,testlen,real)
    best = 10^5
   
    print ("loaded")
    trainset = DataLoader(trainset, batch_size = batch_size, shuffle=True)
    testset = DataLoader(testset, batch_size = testlen, shuffle=True)
    count=0
    for e in range(epoch):
        for batch in trainset:
            optimizer.zero_grad()
            loss = 0

            x, f_x = batch

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
                if accuracy > best:
                    best = accuracy
                    if save:
                        torch.save(model.state_dict(), f"model_sol_{n}-{datalen}-{real}-{comm}.pt")



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


