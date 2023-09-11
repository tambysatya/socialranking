from MIP.problem import random_coalitions, Problem
import random

from lexcell import lex_cell, adv_lex_cell, adv_lex_cell, linear_adv_lex_cell
import numpy as np
import torch




class SetCover ():
    def __init__(self, nitems, nclasses, weights, adjlist): # Int -> Int -> [Int] -> [[Int]]
        self.nitems = nitems
        self.nclasses = nclasses
        self.weights = weights
        self.adjlist = adjlist #the list of items covered by each set

        #the list of sets that covers each items
        self.covers = [set([]) for i in range(nitems)]
        for i, its in enumerate(adjlist):
            for item in its:
                self.covers[item].add(i)

    def toProblem (self):
        ctrs = []
        lb = [1 for i in range(self.nitems)]
        ub = [self.nclasses for i in range(self.nitems)]
        for i in range(self.nitems):
            row = [0 for i in range(self.nclasses)]
            for s in self.covers[i]:
                row[s] = 1
            ctrs.append(row)
        pb = Problem (self.weights, ctrs, ub, lb=lb)
        return pb
    
    def computeRHSChanges(self, coal): 
        """ given a list of sets (the coalition), identifies which items cannot be covered, and relax the corresponding constraint """

        items = set(range(self.nitems))
        covered = set([])
        for s in coal:
            for item in self.adjlist[s]:
                covered.add(item)
        notcovered = items.difference(covered)
        return [(item,0) for item in notcovered]

    def heuristic(self):
        sol = []
        tocover = set(range(self.nitems))
        candidates = set(range(self.nclasses))
        eps=1e-9

        while (tocover != set()):
            best = None
            bestval = None
            for c in candidates:
                covers = tocover.intersection(self.adjlist[c]) #new items covered by the candidates
                val = self.weights[c] / (len(covers)+eps)
                if bestval == None or bestval < val:
                    bestval = val
                    best = c
            sol.append(best)
            candidates.remove(best)
            tocover = tocover.difference(self.adjlist[best])
        score = sum([self.weights[c] for c in sol])
        return score,sol
    def greedy(self, order):
        tocover = set(range(self.nitems))
        sol = []

        for c in order:
            covers = tocover.intersection(self.adjlist[c])
            if covers != set():
                sol.append(c)
                tocover = tocover.difference(self.adjlist[c])
        score = sum([self.weights[c] for c in sol])
        return score
        
            
def generateSC(nitems, nclasses, maxweight, density):
    adjlist = []
    weights=[]
    itemset = set([])
    for i in range(nclasses):
        weights.append(random.randint(-maxweight, -1))
    for i in range (nclasses):
        c = []
        for item in range (nitems):
            rnd = random.uniform(0,1)
            if (rnd <= density):
                c.append(item)
                itemset.add(item)
        adjlist.append(c)
    #translation des indices
    itemmap = {}
    final_adjlist = []
    for i, it in enumerate(itemset):
        itemmap[it] = i
    for c in adjlist:
        final_c = set([])
        for item in c:
            final_c.add(itemmap[item])
        final_adjlist.append(final_c)
    return SetCover(len(itemset), nclasses, weights, final_adjlist)



def test_setcover(nitems, nsets, maxweight, density, l, ncoal, eps, nclasses):
    individuals = set(range(nsets))
    ins = generateSC(nitems, nsets, maxweight, density)
    pb = ins.toProblem()

    opt = pb.solve()[0]

    coals = random_coalitions(individuals, l,ncoal)
    print ("nb_coals=", len(coals))

    evaluation = map(lambda coal: pb.solve_coalition(coal, rhschanges=ins.computeRHSChanges(coal)), coals)
    scores, sols = zip (*evaluation)

    order = adv_lex_cell(individuals, list(zip(sols,coals)), scores, eps)
    adv_lex = ins.greedy(order)

    lin_order = linear_adv_lex_cell(individuals,list (zip (sols,coals)), scores, nclasses)
    lin_lex=ins.greedy(lin_order)



    real =ins.heuristic()[0]
    max_score = torch.tensor(max(scores))

    print (f"opt={opt} geometric={adv_lex} ntiles={lin_lex} heuristic={real} max={max_score}")
    return opt, adv_lex, lin_lex, real, max_score


def run_tests_setcover(nitems,nsets,maxweight, density,l,ncoal,eps,nclasses,ntests=10):
    opttab, geotab, ntilestab, heuristictab, maxscoretab = [],[],[],[],[]
    for i in range(ntests):
       opt, geo, ntiles, heur, maxscore = test_setcover(nitems, nsets, maxweight, density, l, ncoal, eps, nclasses) 
       opttab.append(opt)
       geotab.append(geo)
       ntilestab.append(ntiles)
       heuristictab.append(heur)
       maxscoretab.append(maxscore)
    
    opttab = np.array(opttab)
    geotab = np.array(geotab)
    ntilestab = np.array(ntilestab)
    heuristictab = np.array(heuristictab)
    maxscoretab = np.array(maxscoretab)

    print (f"opt={opttab.mean()} geo={geotab.mean()} ntiles={ntilestab.mean()} heur={heuristictab.mean()} max_coal={maxscoretab.mean()}")


