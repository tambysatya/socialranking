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

from NN.blocks import ResidualLinear, GraphConvolution
from NN.dataset import KPDataset

n=100
m=10
hid=10000
l = 50

datalen=100#test
testlen=100

gpu="cuda:1"
lr=1e-6
comm = f"{gpu}-gcn"

class TestNet (nn.Module):
    def __init__ (self, n, m)
        super (TestNet, self).__init__()

        self.initial_fts = torch.cat((torch.ones(n), torch.zeros(m))).reshape(1,n+m,1)
        self.gcninit = GCNLinear(1, 50)
        self.gcnhid1 = GCNResnetLinear(50)
        self.gcnhid2 = GCNResnetLinear(50)
        self.gcnlast = GCNLinear(50,1)


    def forward (self, x):
        self.train()
        bsize = x.size()[0]
        inputs = self.initial_fts.repeat(bsize,1,1)
        inputs = torch.cat((x,inputs))
        x = self.gcninit1((inputs,x))
        x = F.relu(x)
        x = self.gcinhid1(x)
        x = self.gcnhid2(x)
        y = self.gcnlast(x)
        return y


   
def plot_grads(model):
    total_norm = 0
    for param in model.parameters():
        if (None != param.grad):
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item()**2
    total_norm = total_norm ** 0.5
    return total_norm


def train_gcn(model,epoch, batch_size, real=False):
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
        for batch in dataset:
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
            writer.add_scalar("grad/norm", plot_grads(model), count)
            count += 1
        with torch.no_grad():
            for batch in testset:
                x, f_x = batch
                output = model.forward(x)
                loss = F.binary_cross_entropy_with_logits(output, f_x)
                #loss = lossfunction(testtarget, output)
                writer.add_scalar("loss/test", loss, e)

                output = torch.sigmoid(output)
                output[output>0.5]=1
                output[output<=0.5]=0
                #print (output[0])
                accuracy = (n - (output-f_x).abs().sum(dim=1)).mean()
                writer.add_scalar("loss/accuracy", accuracy , e)
                if accuracy > best:
                    best = accuracy
                    torch.save(model.state_dict(), f"model_sol_{n}.pt")



def show_params(model):
    for param in model.parameters():
        print (param)

        
if __name__ == '__main__':
    if not sys.argv[2:]:
        sys.stdout.write("Size needed in argument : epoch batchsize\n")
        sys.exit(0)

    use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:1" if use_cuda else "cpu")
    device = None
    if use_cuda:
        #device = torch.device(auto_gpu_selection())
        device = torch.device(gpu)
    else:
        device = torch.device("cpu")

    torch.set_default_dtype(torch.float32)

    batchsize = int(sys.argv[2])
    A = torch.load(f"A-{n}_{m}_{l}.pt").to(device)
    objs = torch.load(f"objs-{n}_{m}_{l}.pt").to(device)
    b = torch.load(f"b-{n}_{m}_{l}.pt").to(device)
 
    
    print("Training on " + str(device))
    print("before training")
    model = TestNet(n,m).to(device) # predicts the solution
    train_sol(model,int(sys.argv[1]),int(sys.argv[2]))
    #model = TestNet(objs/1000,A/1000,b/(n*1000), 1).to(device) # predicts the optimal value
    #train_opt(model,int(sys.argv[1]),int(sys.argv[2]),F.mse_loss)
    #show_params(model)
    #train(5,int(1e7))
    print("after training")
    show_params(model)


