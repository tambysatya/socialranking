
from MIP.problem import Problem
import random


# generateCorrelatedKP :: Int -> Int -> (Weights -> IO Cost) -> (Costs, Weights, Cmax)
# une fonction generale pour creer des instances correlees
def generateCorrelatedKP (nbItems, valrange, correlation):
    #from Pisinger D., 2005

    weights = list (map (lambda i: random.randint(1,valrange), range(0,nbItems)) )
    values = list (map (correlation, weights))
    cmax = sum (weights) / 2

    
    return Problem(values, [weights], [cmax])

def generateStronglyCorrelatedKP (nbItems, valrange):
    return generateCorrelatedKP (nbItems, valrange, lambda wi: wi + valrange/10)

def generateWeaklyCorrelatedKP (nbItems, valrange):
    return generateCorrelatedKP (nbItems, valrange, lambda wi: random.randint(wi - valrange/10, wi + valrange/10))

def generateUniformKP (nbItems, valrange):
    return generateCorrelatedKP (nbItems, valrange, lambda wi: random.randint(1,valrange))




        




