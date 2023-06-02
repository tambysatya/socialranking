
from MIP.kp import generateUniformKPND, generateWeaklyCorrelatedKPND, generateStronglyCorrelatedKPND, kp_greedy
from MIP.kp import rndGenerateUniformKPND
from MIP.problem import random_coalitions, Problem
from lexcell import lex_cell, adv_lex_cell
from operator import itemgetter
import random
import numpy as np
import torch
from tqdm import tqdm

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

datalen = 100000 #nb datas generated
n=50
l = 25
nd=10
ncoal = 100


def generate_mats (n,l,nd,ncoal):
    individuals = set(range(n))
    kp = rndGenerateUniformKPND(n,1000,nd)
    mat = kp.toAdjacencyMatrix()
    coals = random_coalitions(individuals, l,ncoal)
    scores,sols = zip(*(list (map (kp.solve_coalition, coals))))

    order = adv_lex_cell(individuals, list(zip(sols,coals)), scores)

    #tgt = torch.zeros(n)
    #for i,vi in enumerate(order):
    #    tgt[vi]=i

    opt,real_sol = kp.solve()

    adv_lex_greed, lex_sol = kp.greedy(order)

    real_order = kp.trivial_order()
    real_greed, _ =kp.greedy(real_order)


    return mat, lex_sol, torch.tensor(real_sol), adv_lex_greed, real_greed, opt
    
def generate_gcn_dataset():
    dataset=[]
    lex_acc= []
    greed_acc = []

    for i in tqdm(range(datalen)):
        ret = generate_mats(n,l,nd,ncoal)
        lex_acc.append(ret[3]/ret[5])
        greed_acc.append(ret[4]/ret[5])
        dataset.append(ret)

    mats, lex_tgts, real_tgts, adv_greed, reals, opts = list(zip(*dataset))
    matsset = torch.stack(mats)
    lex_tgts_dset = torch.stack(lex_tgts)
    real_tgts_dset = torch.stack(real_tgts)
    adv_vals = torch.tensor(adv_greed).reshape(datalen,1)
    opt_vals = torch.tensor(opts).reshape(datalen,1)

    print ("accuracy=", np.array(lex_acc).mean(), " greedy=", np.array(greed_acc).mean())

    torch.save(matsset,f"adjmat-{n}_{nd}_{l}_{datalen}.pt")
    torch.save(lex_tgts_dset,f"lex_tgts-{n}_{nd}_{l}_{datalen}.pt")
    torch.save(real_tgts_dset,f"real_tgts-{n}_{nd}_{l}_{datalen}.pt")
    torch.save(adv_vals,f"adv_vals-{n}_{nd}_{l}_{datalen}.pt")
    torch.save(opt_vals,f"opt_vals-{n}_{nd}_{l}_{datalen}.pt")






def generate_testset (n,l,nd,ncoal):
    individuals = set(range(n))
    objcoefs = torch.load(f"objs-{n}_{nd}_{l}.pt").tolist()
    A= torch.load(f"A-{n}_{nd}_{l}.pt").tolist()
    b= torch.load(f"b-{n}_{nd}_{l}.pt").tolist()
    kp = Problem(objcoefs, A, b)
    coals = random_coalitions(individuals, l,ncoal)
    scores,sols = zip(*(list (map (kp.solve_coalition, coals))))

    objs,A,b = kp.toTensor() 

    coals = map (kp.tensorCoalition, coals)
    inputs = list (map (lambda ci, si, soli: torch.cat([ci,torch.tensor([si]),torch.tensor(soli)],dim=0), coals, scores, sols))

    ret = torch.stack(inputs)
    torch.save(ret,f"testset-{n}_{nd}_{l}.pt")
 
def generate_dataset (n,l,nd,ncoal):
    individuals = set(range(n))
    kp = rndGenerateUniformKPND(n,1000,nd)
    coals = random_coalitions(individuals, l,ncoal)
    scores,sols = zip(*(list (map (kp.solve_coalition, coals))))

    objs,A,b = kp.toTensor() 

    coals = map (kp.tensorCoalition, coals)
    inputs = list (map (lambda ci, si, soli: torch.cat([ci,torch.tensor([si]),torch.tensor(soli)],dim=0), coals, scores, sols))

    ret = torch.stack(inputs)
    torch.save(ret,f"dataset-{n}_{nd}_{l}.pt")
    torch.save(A,f"A-{n}_{nd}_{l}.pt")
    torch.save(objs,f"objs-{n}_{nd}_{l}.pt")
    torch.save(b, f"b-{n}_{nd}_{l}.pt")
    
    

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

    real_order = kp.trivial_order()
    real =kp.greedy(real_order)
    real_order = kp.trivial_order()
    real_order.reverse()
    rev_real = kp.greedy(real_order)


    #real_greedy_order = kp_greedy(kp.objcoefs, kp.A[0])
    #real_greedy = kp.greedy(real_greedy_order)
    #print ("opt=",opt," adv_lex=", adv_lex, " lex=", lex, " adv_rev_lex=", adv_rev_lex, " rev_lex=",rev_lex, " rnd=", rnd, " real_greed=", real_greedy)
    print ("opt=",opt," adv_lex=", adv_lex, " lex=", lex, " adv_rev_lex=", adv_rev_lex, " rev_lex=",rev_lex, " rnd=", rnd, " real=", real, " rev_real=", rev_real)

    return opt, adv_lex, lex, rnd, real


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


#generate_dataset (n,l,nd,ncoal)
#generate_testset (n,l,nd,100)

def test():
    opttab, advtab, lextab, rndtab, realtab =[],[],[],[],[]
    for i in range(10):
        opt,adv_lex, lex, rnd, real = adv_test_kpnd(n, 10)
        opttab.append(opt)
        advtab.append((adv_lex/opt)*100)
        lextab.append((lex/opt)*100)
        rndtab.append((rnd/opt)*100)
        realtab.append((real/opt)*100)

    opttab=np.array(opttab)
    advtab=np.array(advtab)
    lextab=np.array(lextab)
    rndtab=np.array(rndtab)
    realtab = np.array(realtab)

    print ("opt=",opttab.mean(), " advtab=", advtab.mean(), " lex=", lextab.mean(), " rnd=", rndtab.mean(), " real=", realtab.mean())


if __name__ == '__main__':
    generate_gcn_dataset()
#test()
