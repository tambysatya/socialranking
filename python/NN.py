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

n=100
m=10
hid=10000

gpu="cuda:0"
lr=1e-6
comm = f"{gpu}-sol"

class ResidualLinear (nn.Module):
    def __init__(self, in_features, bias=True):
        super (ResidualLinear, self).__init__()
        self.linear1 = nn.Linear(in_features, in_features, bias=bias)
        self.bn = nn.BatchNorm1d(in_features)
        self.linear2 = nn.Linear(in_features, in_features, bias=bias)
    def forward (self, data):
        x = data
        res = self.bn(x)
        res = self.linear1(res)
        res = F.relu(res)
        res = self.linear2(res)
        x = x + res
        x = F.relu(x)
        return x

class TestNet (nn.Module):
    def __init__ (self, objs, A, b, outputsize): 
        super (TestNet, self).__init__()
        

        self.instance = torch.cat((objs,A.reshape(n*m),b))
        in_features = n+n*m+m+n
        self.linear1 = nn.Linear(n+n*m+m+n,hid, bias=True) #f(x) = ax+b
        self.b1 = ResidualLinear(hid)
        self.b2 = ResidualLinear(hid)
        self.linear2 = nn.Linear(hid,outputsize, bias = True)
        #self.linear2 = nn.Linear(hid,n+1, bias = True)



    def forward (self, x):
        self.train()
        bsize = x.size()[0]
        inputs = torch.cat((self.instance.repeat(bsize,1),x), dim=1)
        x = self.linear1(inputs)
        x = F.relu(x)
        x = self.b1(x)
        x = self.b2(x)
        y = self.linear2(x)
        return y


   
def plot_grads(model):
    total_norm = 0
    for param in model.parameters():
        if (None != param.grad):
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item()**2
    total_norm = total_norm ** 0.5
    return total_norm

def train_sol(model,epoch, batch_size):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    writer = SummaryWriter(comment=f"n-100_p-10_rnd-{comm}")
    dataset = torch.load("dataset.pt")
    testset = torch.load("testset.pt")
    best = 10^5
   
    print ("loaded")
    dataset = DataLoader(dataset, batch_size = batch_size, shuffle=True)

    testinput = testset[:,:n].to(device)
    #testtarget = testset[:,n:].to(device)
    #testtarget[:,n] = testtarget[:,n]/100000
    testtarget = testset[:,n+1:].to(device)
    
    count=0
    for e in range(epoch):
        for batch in dataset:
            optimizer.zero_grad()
            loss = 0
            x = batch[:,:n]

            #only sol:
            target = batch[:,n+1:].to(device)

            x = x.to(device)
            y = model.forward(x)

            loss = F.binary_cross_entropy_with_logits(y, target)
            loss.backward()
            optimizer.step() # un pas de la descente de gradient

            writer.add_scalar("loss/train", loss, count)
            writer.add_scalar("grad/norm", plot_grads(model), count)
            count += 1
        with torch.no_grad():
            output = model.forward(testinput)
            loss = F.binary_cross_entropy_with_logits(output, testtarget)
            #loss = lossfunction(testtarget, output)
            writer.add_scalar("loss/test", loss, e)

            output = torch.sigmoid(output)
            output[output>0.5]=1
            output[output<=0.5]=0
            #print (output[0])
            accuracy = (n - (output-testtarget).abs().sum(dim=1)).mean()
            writer.add_scalar("loss/accuracy", accuracy , e)
            if accuracy > best:
                best = accuracy
                torch.save(model.state_dict(), "model_sol.pt")

def train_opt(model,epoch, batch_size, lossfun):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    writer = SummaryWriter(comment=f"n-100_p-10_rnd-{comm}")
    dataset = torch.load("dataset.pt")
    testset = torch.load("testset.pt")
   
    print ("loaded")
    dataset = DataLoader(dataset, batch_size = batch_size, shuffle=True)

    testinput = testset[:,:n].to(device)
    testtarget = testset[:,n].to(device).reshape(-1,1)

    best = 1e-7
    
    count=0
    for e in range(epoch):
        model.train()
        for batch in dataset:
            optimizer.zero_grad()
            loss = 0
            x = batch[:,:n]

            #only opt:
            target = batch[:,n].to(device).reshape(-1,1)

            x = x.to(device)
            y = model.forward(x)

            loss = lossfun(y, target)
            loss.backward()
            optimizer.step() # un pas de la descente de gradient

            writer.add_scalar("loss/train", loss, count)
            writer.add_scalar("grad/norm", plot_grads(model), count)
            count += 1
        with torch.no_grad():
            model.test()
            output = model.forward(testinput)
            loss = lossfun(output, testtarget)
            #loss = lossfunction(testtarget, output)
            if loss < best:
                best = loss
                torch.save(model.state_dict(), "model_opt.pt")
            writer.add_scalar("loss/test", loss, e)
        


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
    A = torch.load("A.pt").to(device)
    objs = torch.load("objs.pt").to(device)
    b = torch.load("b.pt").to(device)
 
    
    print("Training on " + str(device))
    print("before training")
    model = TestNet(objs/1000,A/1000,b/100000, 100).to(device) # predicts the solution
    train_sol(model,int(sys.argv[1]),int(sys.argv[2]))
    #model = TestNet(objs/1000,A/1000,b/(n*1000), 1).to(device) # predicts the optimal value
    #train_opt(model,int(sys.argv[1]),int(sys.argv[2]),F.mse_loss)
    #show_params(model)
    #train(5,int(1e7))
    print("after training")
    show_params(model)


