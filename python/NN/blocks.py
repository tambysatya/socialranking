import torch
#import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
#import torch_optimizer as optim
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader



torch.set_default_dtype(torch.float32)


class GraphConvolution(Module):
#code from https://github.com/tkipf/pygcn
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, data):
        input, adj = data
        #edited to make batch multiplication
        support = torch.matmul(input, self.weight)
        output = torch.bmm(adj, support)
        if self.bias is not None:
            return output + self.bias, adj
        else:
            return output, adj

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class ResidualLinear (nn.Module):
    def __init__(self, in_features, bias=True):
        super (ResidualLinear, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        self.linear1 = nn.Linear(in_features, in_features, bias=bias)
        self.linear2 = nn.Linear(in_features, in_features, bias=bias)

        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()

    def forward (self, data):
        x = data
        res = self.bn1(x)
        res = F.relu(res)
        res = self.linear1(res)
        res = self.dropout1(res)

        res = self.bn2(res)
        res = F.relu(res)
        res = self.linear2(res)
        res = self.dropout2(res)
        x = x + res
        #x = F.relu(x)
        return x

