
from MIP.kp import generateUniformKPND, generateWeaklyCorrelatedKPND, generateStronglyCorrelatedKPND, kp_greedy
from MIP.kp import rndGenerateUniformKPND
from lexcell import lex_cell, random_coalitions, adv_lex_cell
from operator import itemgetter
import random
import numpy as np

#f= [1,2]
#A = [[2,3],[2,5]]
#bs = [3,3]
#
#p = Problem (f,A,bs)
#
#print (p.solve())
#
#print (p.pb.variables.get_upper_bounds())
#print (p.solve_coalition({1}))
#print (p.pb.variables.get_upper_bounds())

#kp = generateUniformKP(10,20)
#print(kp.solve())
#print(kp.solve_coalition({2,5,8,9}))
#
#order = [2,5]
#
#print (kp.display())
#print (kp.greedy(order))

n=1000
l = 100
ncoal = 100000

def adv_test_kpnd(n_individuals, nd):
    individuals = set(range(n_individuals))
    kp = rndGenerateUniformKPND(n_individuals,1000, nd)
    #kp = generateUniformKPND(n_individuals,1000, nd)
    #kp = generateWeaklyCorrelatedKPND(n_individuals,1000, nd)
    #kp = generateStronglyCorrelatedKPND(n_individuals,1000, nd)
    coals = random_coalitions(individuals, l,ncoal)
    print ("nb_coals=", len(coals))
    scores,sols = zip(*(list (map (kp.solve_coalition, coals))))

    order = adv_lex_cell(individuals, list(zip(sols,coals)), scores)

    opt = kp.solve()[0]
    adv_lex = kp.greedy(order)
    order = adv_lex_cell(individuals,list (zip (sols,coals)), scores)
    order.reverse()
    adv_rev_lex = kp.greedy(order)

    order = lex_cell(individuals,coals, scores)

    opt = kp.solve()[0]
    lex = kp.greedy(order)
    order = lex_cell(individuals,coals, scores)
    order.reverse()
    rev_lex = kp.greedy(order)



    rnd_order = list(range(n_individuals))
    random.shuffle(rnd_order)
    rnd = kp.greedy(rnd_order)


    #real_greedy_order = kp_greedy(kp.objcoefs, kp.A[0])
    #real_greedy = kp.greedy(real_greedy_order)
    #print ("opt=",opt," adv_lex=", adv_lex, " lex=", lex, " adv_rev_lex=", adv_rev_lex, " rev_lex=",rev_lex, " rnd=", rnd, " real_greed=", real_greedy)
    print ("opt=",opt," adv_lex=", adv_lex, " lex=", lex, " adv_rev_lex=", adv_rev_lex, " rev_lex=",rev_lex, " rnd=", rnd)

    return opt, adv_lex, lex, rnd


def test_kpnd(n_individuals, nd):
    individuals = set(range(n_individuals))
    kp = generateStronglyCorrelatedKPND(n_individuals,1000, nd)
    coals = random_coalitions(individuals, 20, 1000)
    scores = list (map (itemgetter(0),map (kp.solve_coalition, coals)))

    order = lex_cell(individuals,coals, scores)

    opt = kp.solve()[0]
    lex = kp.greedy(order)
    order = lex_cell(individuals,coals, scores)
    order.reverse()
    rev_lex = kp.greedy(order)


    rnd_order = list(range(n_individuals))
    random.shuffle(rnd_order)
    rnd = kp.greedy(rnd_order)


    print ("opt=",opt," lex=", lex, " rev_lex=", rev_lex, " rnd=", rnd)


def test_kp(n_individuals):
    individuals = set(range(n_individuals))
    #kp = generateStronglyCorrelatedKPND(n_individuals,1000, 1)
    #kp = generateWeaklyCorrelatedKP(100,1000) #juste avant
    kp = generateUniformKP (100,1000)
    real_greedy_order = kp_greedy(kp.objcoefs, kp.A[0])
    real_greedy = kp.greedy(real_greedy_order)
    coals = random_coalitions(individuals, 20, 5000)
    scores = list (map (itemgetter(0),map (kp.solve_coalition, coals)))

    order = lex_cell(individuals,coals, scores)
    #print ("scores=",scores)
    #print ("order=",order)

    opt = kp.solve()[0]
    lex = kp.greedy(order)
    #order = lex_cell(individuals,coals, scores)
    #order.reverse()
    rev_lex = "not_computed" #kp.greedy(order)


    rnd_order = list(range(n_individuals))
    random.shuffle(rnd_order)
    rnd = kp.greedy(rnd_order)


    print ("opt=",opt," lex=", lex, " rev_lex=", rev_lex, " real=", real_greedy, " rnd=", rnd)


opttab, advtab, lextab, rndtab =[],[],[],[]
for i in range(10):
    opt,adv_lex, lex, rnd = adv_test_kpnd(n, 10)
    opttab.append(opt)
    advtab.append((adv_lex/opt)*100)
    lextab.append((lex/opt)*100)
    rndtab.append((rnd/opt)*100)

opttab=np.array(opttab)
advtab=np.array(advtab)
lextab=np.array(lextab)
rndtab=np.array(rndtab)

print ("opt=",opttab.mean(), " advtab=", advtab.mean(), " lex=", lextab.mean(), " rnd=", rndtab.mean())



