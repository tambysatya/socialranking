
from MIP.kp import generateUniformKPND, generateWeaklyCorrelatedKPND, generateStronglyCorrelatedKPND, kp_greedy
from MIP.kp import rndGenerateUniformKPND, rndGenerateCorrelatedKPNDbiased
from MIP.problem import random_coalitions, Problem
from lexcell import lex_cell, adv_lex_cell
from operator import itemgetter
import itertools
import random
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

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

if __name__ == '__main__':
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
    
    
def test_kp_density(n_individuals,l, nd, eps):
    individuals = set(range(n_individuals))
    #kp = rndGenerateUniformKPND(n_individuals,10, nd)
    #kp = rndGenerateUniformKPND(n_individuals,1000, nd)
    #kp = generateUniformKPND(n_individuals,1000, nd)
    kp = rndGenerateCorrelatedKPNDbiased(n_individuals, 1000, nd, lambda weights: random.randint(int(weights.sum()/7), int(6*weights.sum()/7)))
    #kp = generateWeaklyCorrelatedKPND(n_individuals,1000, nd)
    #kp = generateStronglyCorrelatedKPND(n_individuals,1000, nd)
    coals = []
    #coals = list(itertools.combinations(individuals, l))
    for i in range (1,l+1):
        combs = itertools.combinations(individuals, i)
        coals += combs
    scores = list (map (kp.density, coals))
    print ("nb_coals=", len(coals))
    scores = list (map (kp.density_scaled, coals))

    order = lex_cell(individuals,coals, scores, eps)

    opt = kp.solve()[0]
    lex,_ = kp.greedy(order)
    order = lex_cell(individuals,coals, scores, eps)
    order.reverse()
    rev_lex,_ = kp.greedy(order)



    rnd_order = list(range(n_individuals))
    random.shuffle(rnd_order)
    rnd,_ = kp.greedy(rnd_order)

    real_order = kp.trivial_order()
    real,_ =kp.greedy(real_order)
    real_order = kp.trivial_order()
    real_order.reverse()
    rev_real,_ = kp.greedy(real_order)

    scaled_order = kp.trivial_order_scaled()
    scaled, _ = kp.greedy(scaled_order)

    max_score = torch.tensor(max(scores))

    print ("opt=",opt," density lex=", lex, " rev_lex=",rev_lex, " scaled_real=", scaled, " real=", real, " rev_real=", rev_real, " max_coals=", max_score)
    #print ("opt=",opt," density lex=", lex, " rev_lex=",rev_lex, " rnd=", rnd, " real=", real, " rev_real=", rev_real, " max_coals=", max_score)

    return opt,  lex, scaled, real, max_score



def adv_test_kpnd(n_individuals,l, ncoal, nd, eps):
    individuals = set(range(n_individuals))
    #kp = rndGenerateUniformKPND(n_individuals,10, nd)
    #kp = rndGenerateUniformKPND(n_individuals,1000, nd)
    #kp = generateUniformKPND(n_individuals,1000, nd)
    #kp = rndGenerateCorrelatedKPNDbiased(n_individuals, 1000, nd, lambda weights: random.randint(int(weights.sum()/4), int(3*weights.sum()/4)))
    kp = rndGenerateCorrelatedKPNDbiased(n_individuals, 1000, nd, lambda weights: random.randint(int(weights.sum()/7), int(6*weights.sum()/7)))
    #kp = generateWeaklyCorrelatedKPND(n_individuals,1000, nd)
    #kp = generateStronglyCorrelatedKPND(n_individuals,1000, nd)
    coals = random_coalitions(individuals, l,ncoal)
    print ("nb_coals=", len(coals))
    scores,sols = zip(*(list (map (kp.solve_coalition, coals))))

    order = adv_lex_cell(individuals, list(zip(sols,coals)), scores, eps)

    opt = kp.solve()[0]
    adv_lex,_ = kp.greedy(order)
    order = adv_lex_cell(individuals,list (zip (sols,coals)), scores, eps)

    order.reverse()
    adv_rev_lex,_ = kp.greedy(order)

    order = lex_cell(individuals,coals, scores, eps)

    opt = kp.solve()[0]
    lex,_ = kp.greedy(order)
    order = lex_cell(individuals,coals, scores, eps)
    order.reverse()
    rev_lex,_ = kp.greedy(order)



    #rnd_order = list(range(n_individuals))
    #random.shuffle(rnd_order)
    #rnd,_ = kp.greedy(rnd_order)

    scaled_order = kp.trivial_order_scaled()
    scaled, _ = kp.greedy(scaled_order)

    real_order = kp.trivial_order()
    real,_ =kp.greedy(real_order)
    real_order = kp.trivial_order()
    real_order.reverse()
    rev_real,_ = kp.greedy(real_order)

    max_score = torch.tensor(max(scores))

    print ("opt=",opt," adv_lex=", adv_lex, " lex=", lex, " adv_rev_lex=", adv_rev_lex, " rev_lex=",rev_lex, " scaled=", scaled, " real=", real, " rev_real=", rev_real, " max_coals=", max_score)

    return opt, adv_lex, lex, scaled, real, max_score


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

def test_opt_score(n, l, ncoal, nd, eps, ntests=10):
    opttab, advtab, lextab, scaledtab, realtab =[],[],[],[],[]
    max_coals = []
    for i in range(ntests):
        opt,adv_lex, lex, scaled, real, max_coal = adv_test_kpnd(n,l, ncoal,nd, eps)
        opttab.append(opt)
        advtab.append((adv_lex/opt)*100)
        lextab.append((lex/opt)*100)
        scaledtab.append((scaled/opt)*100)
        realtab.append((real/opt)*100)
        max_coals.append((max_coal/opt)*100)

    opttab=np.array(opttab)
    advtab=np.array(advtab)
    lextab=np.array(lextab)
    scaledtab=np.array(scaledtab)
    realtab = np.array(realtab)
    max_coals = np.array(max_coals)

    print ("opt=",opttab.mean(), " advtab=", advtab.mean(), " lex=", lextab.mean(), " scaled=", scaledtab.mean(), " real=", realtab.mean(), " max_coals=", max_coals.mean())
    return advtab.mean(), lextab.mean(), scaledtab.mean(), realtab.mean(), max_coals.mean()

def test_density_score(n, l, nd, eps, ntests=10):
    opttab,  lextab, scaledtab, realtab =[],[],[],[]
    max_coals = []
    for i in range(ntests):
        opt, lex, scaled, real, max_coal = test_kp_density(n,l, nd, eps)
        opttab.append(opt)
        lextab.append((lex/opt)*100)
        scaledtab.append((scaled/opt)*100)
        realtab.append((real/opt)*100)
        max_coals.append((max_coal/opt)*100)

    opttab=np.array(opttab)
    lextab=np.array(lextab)
    scaledtab=np.array(scaledtab)
    realtab = np.array(realtab)
    max_coals = np.array(max_coals)

    print ("density opt=",opttab.mean(), " lex=", lextab.mean(), " scaled=", scaledtab.mean(), " real=", realtab.mean(), " max_coals=", max_coals.mean())
    return  lextab.mean(), scaledtab.mean(), realtab.mean(), max_coals.mean()


def plot_test(n,ls,ncoals,nd, eps, ntests, test_fun):
    #points = []
    file = open("figures/logs.txt","a")
    for l in ls:
        points_l = []
        for ncoal in ncoals:
           advmean, lexmean, rndmean, greedmean, maxcoalmean = test_fun(n,l,ncoal,nd, eps, ntests=ntests) 
           points_l.append(advmean)
           file.write(f"{n};{l};{ncoal};{advmean};{lexmean};{rndmean};{greedmean}\n")
        #points.append(points_l)
        plt.plot(ncoals, points_l, label=f"{l}")
    plt.xlabel("nb coalitions")
    plt.ylabel("% accuracy")
    legend_outside = plt.legend(bbox_to_anchor=(1.05, 1.0), 
                            loc='upper left')
    plt.savefig(f"figures/fig_n-{n}_eps-{eps}.png",
            dpi=150, 
            format='png', 
            bbox_extra_artists=(legend_outside,), 
            bbox_inches='tight')
    plt.clf()
    file.close()

    

if __name__ == '__main__':
    #generate_gcn_dataset()
    #plot_test(50,[15,25], [100,200],10, 0.1, 10,test_opt_score)
    #plot_test(50,[5,15,25,35], [10,50,100,150,200,500,1000],10)
    #plot_test(50,[5,10,15,20,25,30,35], [10,50,100,150,200,500,1000],10)
    #plot_test(100,[5,25,50,75], [10,50,100,150,200,500,1000],10)
    #plot_test(1000,[50,250,500,750], [10,50,100,150,200,500,1000],10)
########## window laurent
    #test_opt_score(50,25, 500,10, 0.1, ntests=10)  #opt= 12013.4  advtab= 97.0256  lex= 91.87744  scaled= 87.56992  real= 82.68021  max_coals= 93.15741
    #test_opt_score(50,15, 500,10, 0.1, ntests=10)   #opt= 11547.1  advtab= 87.34828  lex= 81.81285  scaled= 94.34419  real= 88.75807  max_coals= 77.76577
    test_opt_score(100,50, 1000,10, 0.1, ntests=100)  


######### mon approche
    #test_opt_score(100,50, 500,10, 0.1, ntests=10) # opt= 22987.3  advtab= 94.50146  lex= 80.72317  scaled= 92.069115  real= 83.20607  max_coals= 88.2104
    #test_opt_score(100,50, 1000,10, 0.1, ntests=10) # opt= 25777.8  advtab= 94.43021  lex= 88.21464  scaled= 91.421  real= 84.35704  max_coals= 87.06607


    #test_opt_score(50,25, 500,10, 0.1, ntests=10) #  opt= 10200.7  advtab= 96.77643  lex= 88.79738  scaled= 91.531715  real= 84.78645  max_coals= 94.44302
    #test_opt_score(50,25, 1000,10, 0.1, ntests=10) # opt= 11495.8  advtab= 97.37021  lex= 93.06477  scaled= 93.59407  real= 89.80574  max_coals= 92.752785
    #test_opt_score(50,25, 1000,10, 0.05, ntests=10) # opt= 10917.5  advtab= 95.33132  lex= 88.8538  scaled= 86.91655  real= 82.518776  max_coals= 94.22127
    #test_opt_score(50,25, 1500,10, 0.1, ntests=10) # opt= 10603.3  advtab= 97.82802  lex= 91.16337  scaled= 92.175026  real= 86.21991  max_coals= 94.95763
    #test_opt_score(50,25, 1500,10, 0.05, ntests=10) # opt= 10036.8  advtab= 98.06808  lex= 92.48436  scaled= 94.9505  real= 86.625725  max_coals= 96.46199


    #test_density_score(100,2, 10, 0, ntests=100) 
    #test_density_score(50,2, 10, 0, ntests=100)  # density opt= 11100.4  lex= 93.45003  scaled= 93.68198  real= 86.89389  max_coals= 27.208838

    #test(500,250,2000,10)
    #test(50,35,1000,nd)
