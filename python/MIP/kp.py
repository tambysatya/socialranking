
from MIP.problem import Problem, random_coalitions, random_coalitions_distrib
import random
from operator import itemgetter
import torch
import numpy as np
import itertools

from lexcell import lex_cell, linear_lex_cell, adv_lex_cell, adv_lex_cell, linear_adv_lex_cell

def compute_scores (problem):
    profits = problem.objcoefs
    ret = []
    for i in range (len(profits)):
        w = 0
        for row in problem.A:
            w += row[i]
        ret.append(profits[i]/w)
    return ret

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
    return generateCorrelatedKPND (nbItems, valrange, nd, lambda wi: random.randint(max(1,wi - valrange/10), wi + valrange/10))

def generateUniformKPND (nbItems, valrange, nd):
    return generateCorrelatedKPND (nbItems, valrange, nd, lambda wi: random.randint(1,valrange))



def kp_greedy (objcoefs, weights):
    individuals = list (range(len(objcoefs)))
    scores = list(torch.tensor(objcoefs) / torch.tensor(weights))

    order = map (itemgetter(1),sorted(zip (scores, individuals), reverse=True))
    return list(order)

def rndGenerateCorrelatedKPND (nbItems, valrange,n, correlation):
    """ random maximal capacity """

    #values = list (map (correlation, weights))
    values = [random.randint (1,valrange) for i in range (nbItems)]
    weights = []
    bs = []
    for i in range(n):
        coefs = list (map (correlation, values))
        weights.append(coefs)
        #bi = random.randint (int(sum(coefs)/2), int(sum(coefs)))
        #bi = random.randint (min(coefs), int(sum(coefs)))
        #bs.append(bi)
    Acoefs = np.array(weights)
    for i in range (n):
        bi = random.randint (Acoefs.max(), int(Acoefs[i].sum()))
        bs.append(bi)

        
    return Problem(values, weights, bs)
def rndGenerateUniformKPND (nbItems, valrange, nd):
    return rndGenerateCorrelatedKPND (nbItems, valrange, nd, lambda wi: random.randint(1,valrange))

def rndGenerateCorrelatedKPNDbiased (nbItems, valrange,n, rhsfun):
    """ biased random maximal capacity """

    #values = list (map (correlation, weights))
    values = [random.randint (1,valrange) for i in range (nbItems)]
    weights = []
    bs = []
    for i in range(n):
        coefs = list (map (lambda _: random.randint(1,valrange), values))
        weights.append(coefs)
        #bi = random.randint (int(sum(coefs)/2), int(sum(coefs)))
        #bi = random.randint (min(coefs), int(sum(coefs)))
        #bs.append(bi)
    Acoefs = np.array(weights)
    for i in range (n):
        bi = rhsfun(Acoefs[i])
        bs.append(bi)

        
    return Problem(values, weights, bs)


def density_test(n_individuals,l, nd, eps, nclasses):
    individuals = set(range(n_individuals))
    #kp = rndGenerateUniformKPND(n_individuals,10, nd)
    kp = rndGenerateUniformKPND(n_individuals,1000, nd)
    #kp = generateUniformKPND(n_individuals,1000, nd)
    #kp = rndGenerateCorrelatedKPNDbiased(n_individuals, 1000, nd, lambda weights: random.randint(int(weights.sum()/4), int(3*weights.sum()/4)))
    #kp = rndGenerateCorrelatedKPNDbiased(n_individuals, 1000, nd, lambda weights: random.randint(int(weights.sum()/7), int(6*weights.sum()/7)))
    #kp = generateWeaklyCorrelatedKPND(n_individuals,1000, nd)
    #kp = generateStronglyCorrelatedKPND(n_individuals,1000, nd)


    #coalition sampling
    coals = []
    #coals = list(itertools.combinations(individuals, l))
    for i in range (1,l+1):
        combs = itertools.combinations(individuals, i)
        coals += combs
    scores = list (map (kp.density, coals))

    #biased_coals = random_coalitions_distrib(list(individuals), l,ncoal, compute_scores(kp)) # sampling  pas uniforme
    print ("nb_coals=", len(coals))
    density_scores = list (map (kp.density_merge, coals))

    opt = kp.solve()[0]

    order = lex_cell(individuals, coals, density_scores, eps)
    adv_lex,_ = kp.greedy(order)

    lin_order = linear_lex_cell(individuals, coals, density_scores, nclasses)
    lin_lex,_= kp.greedy(lin_order)



    scaled_order = kp.trivial_order_scaled()
    scaled, _ = kp.greedy(scaled_order)

    max_score = torch.tensor(max(density_scores))

    print ("opt=",opt," density_geo=", adv_lex, "density_lin=", lin_lex, " scaled=", scaled," max_coals=", max_score)
    return opt, adv_lex,lin_lex, scaled, max_score



def adv_test_kpnd(n_individuals,l, ncoal, nd, eps, nclasses):
    individuals = set(range(n_individuals))
    #kp = rndGenerateUniformKPND(n_individuals,10, nd)
    kp = rndGenerateUniformKPND(n_individuals,1000, nd)
    #kp = generateUniformKPND(n_individuals,1000, nd)
    #kp = rndGenerateCorrelatedKPNDbiased(n_individuals, 1000, nd, lambda weights: random.randint(int(weights.sum()/4), int(3*weights.sum()/4)))
    #kp = rndGenerateCorrelatedKPNDbiased(n_individuals, 1000, nd, lambda weights: random.randint(int(weights.sum()/7), int(6*weights.sum()/7)))
    #kp = generateWeaklyCorrelatedKPND(n_individuals,1000, nd)
    #kp = generateStronglyCorrelatedKPND(n_individuals,1000, nd)


    #coalition sampling
    coals = random_coalitions(individuals, l,ncoal) #uniform
    biased_coals = random_coalitions_distrib(list(individuals), l,ncoal, compute_scores(kp)) # sampling  pas uniforme
    print ("nb_coals=", len(coals))
    density_scores = list (map (kp.density_scaled, coals))
    scores,sols = zip(*(list (map (kp.solve_coalition, coals))))
    biased_scores,biased_sols = zip(*(list (map (kp.solve_coalition, biased_coals))))

    order = lex_cell(individuals, coals, density_scores, eps)
    #order = adv_lex_cell(individuals, list(zip(sols,coals)), scores, eps)

    opt = kp.solve()[0]
    adv_lex,_ = kp.greedy(order)

    lin_order = linear_adv_lex_cell(individuals,list (zip (sols,coals)), scores, nclasses)
    lin_lex,_=kp.greedy(lin_order)
    biased_lin_order = linear_adv_lex_cell(individuals,list (zip (biased_sols,biased_coals)), biased_scores, nclasses) #biased order according to the score from the greedy
    biased_lin,_ = kp.greedy(biased_lin_order)



    order = adv_lex_cell(individuals,list (zip (sols,coals)), scores, eps)
    order.reverse()
    adv_rev_lex,_ = kp.greedy(order)

    order = lex_cell(individuals,coals, scores, eps)

    opt = kp.solve()[0]
    lex,_ = kp.greedy(order)
    order = lex_cell(individuals,coals, scores, eps)
    order.reverse()
    rev_lex,_ = kp.greedy(order)

    scaled_order = kp.trivial_order_scaled()
    scaled, _ = kp.greedy(scaled_order)

    real_order = kp.trivial_order()
    real,_ =kp.greedy(real_order)
    real_order = kp.trivial_order()
    real_order.reverse()
    rev_real,_ = kp.greedy(real_order)

    max_score = torch.tensor(max(scores))

    print ("opt=",opt," adv_lex=", adv_lex, "lin=", lin_lex, "biased_lin=", biased_lin, " lex=", lex, " adv_rev_lex=", adv_rev_lex, " rev_lex=",rev_lex, " scaled=", scaled, " real=", real, " rev_real=", rev_real, " max_coals=", max_score)

    return opt, adv_lex, lin_lex, biased_lin, lex, scaled, real, max_score

   
def test_kps(n, l, ncoal, nd, eps, nclasses, ntests=10):
    print (f"n={n} l={l} ncoal={ncoal} nctrs={nd} eps={eps} nclasses={nclasses}")
    opttab, advtab, lintab, biasedtab, lextab, scaledtab, realtab =[],[],[],[],[],[], []
    max_coals = []
    for i in range(ntests):
        opt,adv_lex, lin_lex, biased_lin,lex, scaled, real, max_coal = adv_test_kpnd(n,l, ncoal,nd, eps, nclasses)
        opttab.append(opt)
        advtab.append((adv_lex/opt)*100)
        lintab.append((lin_lex/opt)*100)
        biasedtab.append((biased_lin/opt)*100)
        lextab.append((lex/opt)*100)
        scaledtab.append((scaled/opt)*100)
        realtab.append((real/opt)*100)
        max_coals.append((max_coal/opt)*100)

    opttab=np.array(opttab)
    advtab=np.array(advtab)
    lintab=np.array(lintab)
    biasedtab = np.array(biasedtab)
    lextab=np.array(lextab)
    scaledtab=np.array(scaledtab)
    realtab = np.array(realtab)
    max_coals = np.array(max_coals)

    print ("opt=",opttab.mean(), " advtab=", advtab.mean(), "lintab=",lintab.mean(), "biased=", biasedtab.mean()," lex=", lextab.mean(), " scaled=", scaledtab.mean(), " real=", realtab.mean(), " max_coals=", max_coals.mean())
    return advtab.mean(), lextab.mean(), scaledtab.mean(), realtab.mean(), max_coals.mean()



        


def tests_density(n,l,nd,eps,nclasses,ntests=10):
    print (f"n={n} l={l} nctrs={nd} eps={eps} nclasses={nclasses}")
    opttab, advtab, lintab, scaledtab = [],[],[],[]
    max_coals = []
    for i in range(ntests):
        opt,adv_lex, lin,scaled, max_coal = density_test(n,l, nd, eps, nclasses)
        opttab.append(opt)
        advtab.append((adv_lex/opt)*100)
        lintab.append((lin/opt)*100)
        scaledtab.append((scaled/opt)*100)
        max_coals.append((max_coal/opt)*100)

    opttab=np.array(opttab)
    advtab=np.array(advtab)
    lintab=np.array(lintab)
    scaledtab=np.array(scaledtab)
    max_coals = np.array(max_coals)

    print ("opt=",opttab.mean(), " advtab=", advtab.mean(), "lintab=",lintab.mean(), " scaled=", scaledtab.mean(), " max_coals=", max_coals.mean())
    return advtab.mean(), scaledtab.mean(), max_coals.mean()


   

