
from MIP.problem import Problem
import random
from operator import itemgetter
import torch


# generateCorrelatedKP :: Int -> Int ->  Int -> (Cost -> IO Weight) -> (Costs, Weights, Cmax)
def generateCorrelatedKPND (nbItems, valrange,n, correlation):
    #from Pisinger D., 2005

    #values = list (map (correlation, weights))
    values = [random.randint (1,valrange) for i in range (nbItems)]
    weights = []
    bs = []
    for i in range(n):
        coefs = list (map (correlation, values))
        #coefs = list (map (lambda i: random.randint(1,valrange), range(0,nbItems)) )
        weights.append(coefs)
        bs.append(sum (coefs) / 2)

    return Problem(values, weights, bs)


def generateStronglyCorrelatedKPND (nbItems, valrange, nd):
    return generateCorrelatedKPND (nbItems, valrange,nd, lambda wi: wi + valrange/10)

def generateWeaklyCorrelatedKPND (nbItems, valrange, nd):
    return generateCorrelatedKPND (nbItems, valrange, nd, lambda wi: random.randint(wi - valrange/10, wi + valrange/10))

def generateUniformKPND (nbItems, valrange, nd):
    return generateCorrelatedKPND (nbItems, valrange, nd, lambda wi: random.randint(1,valrange))



def kp_greedy (objcoefs, weights):
    individuals = list (range(len(objcoefs)))
    scores = list(torch.tensor(objcoefs) / torch.tensor(weights))

    order = map (itemgetter(1),sorted(zip (scores, individuals), reverse=True))
    return list(order)

    


        




