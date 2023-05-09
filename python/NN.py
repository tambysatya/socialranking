#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import sys,os,math,subprocess
import torch
import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

class TestNet (nn.Module):
    def __init__ (self): 
        super (TestNet, self).__init__()
        
        self.linear = nn.Linear(1,50, bias=True) #f(x) = ax+b
        self.linear2 = nn.Linear(50,1, bias = True)


    def forward (self, x):
        x = self.linear(x)
        x = F.relu(x)
        y = self.linear2(x)
        return y


    def infer (self, batch):
        x, f_x = batch
        y = self.forward(x)
        loss = F.l1_loss(f_x, y)
        return loss

   
def generate_dataset(size):
    ret = []
    for i in range(size):
        x = torch.rand([1,1])
        f_x = x**0.5  + 4
        #f_x = x + 4
        ret.append([x,f_x])
    return torch.tensor(ret)
def plot_grads(model):
    total_norm = 0
    for param in model.parameters():
        if (None != param.grad):
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item()**2
    total_norm = total_norm ** 0.5
    return total_norm

def train(model,epoch, size,lossfunction=F.l1_loss):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    writer = SummaryWriter(comment=" olivier")
    dataset = generate_dataset(size)
    batch_size = 100
    
    dataset = DataLoader(dataset, batch_size = batch_size, shuffle=True)
    
    count=0
    for _ in range(epoch):
        for batch in dataset:
            loss = 0
            # loss = model.learn(optimizer, dataset)
            x, f_x = batch[:,0].reshape([batch_size, 1]), batch[:,1].reshape([batch_size,1])
            x = x.to(device)
            f_x = f_x.to(device)
            y = model.forward(x)

            #loss = F.l1_loss(f_x, y)
            #loss = F.mse_loss(f_x, y)
            loss = lossfunction(f_x, y)
            loss.backward()
            optimizer.step() # un pas de la descente de gradient

            writer.add_scalar("loss/train", loss, count)
            writer.add_scalar("grad/norm", plot_grads(model), count)
            count += 1


def show_params(model):
    for param in model.parameters():
        print (param)

def auto_gpu_selection(usage_max=0.01, mem_max=0.05):
#Auto set CUDA_VISIBLE_DEVICES for gpu
#:param mem_max: max percentage of GPU utility
#:param usage_max: max percentage of GPU memory
#:return:

 device = "cpu"
 os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
 if (not os.path.exists("/usr/bin/nvidia-smi")) : return device
 log = str(subprocess.check_output("nvidia-smi", shell=True)).split(r"\n")[6:-1]
 gpu = 0

 for i in range(len(tf.config.list_physical_devices('GPU'))):
    idx = i*4 + 3   
    if idx > log.__len__()-1:
        break
    inf = log[idx].split("|")
    if inf.__len__() < 3:
        break
    usage = int(inf[3].split("%")[0].strip())
    mem_now = int(str(inf[2].split("/")[0]).strip()[:-3])
    mem_all = int(str(inf[2].split("/")[1]).strip()[:-3])
    # print("GPU-%d : Usage:[%d%%]" % (gpu, usage))
    if usage < 100*usage_max and mem_now < mem_max*mem_all:
        device = "cuda:" + str(gpu)
        print("\nAuto choosing vacant GPU-%d : Memory:[%dMiB/%dMiB] , GPU-Util:[%d%%]\n" %  (gpu, mem_now, mem_all, usage))
        return device
    #print("GPU-%d is busy: Memory:[%dMiB/%dMiB] , GPU-Util:[%d%%]" %
     #     (gpu, mem_now, mem_all, usage))
    gpu += 1

 print("\nNo vacant GPU, use CPU instead\n")
 return device # set to cpu

        
if __name__ == '__main__':
    if not sys.argv[2:]:
        sys.stdout.write("Size needed in argument : epoch datasetsize\n")
        sys.exit(0)

    #use_cuda = torch.cuda.is_available()
    #device = torch.device("cuda:1" if use_cuda else "cpu")
    device = torch.device(auto_gpu_selection())

    torch.set_default_dtype(torch.float32)
    
    model = TestNet().to(device)    
    print("before training")
    #show_params(model)
    #train(5,int(1e7))
    print("Training on " + str(device))
    train(model,int(sys.argv[1]),int(sys.argv[2]), F.mse_loss)
    print("after training")
    show_params(model)


