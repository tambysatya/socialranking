
from MIP.problem import Problem
import random
import numpy as np



class IndependentSet():

    def __init__ (self, nvertices, weights, adjlist):
        self.nvertices = nvertices
        self.weights = weights
        self.adjlist = adjlist
    
    def toProblem (self):
        
        ctrs = []
        b = []
        for x, y in self.adjlist:
            #moche
            row = [1 if i == x or i == y else 0 for i in range (self.nvertices)]
            ctrs.append (row)

            b.append(1)
        return Problem (self.weights, ctrs, b)

    def neighbors (self, x):
        """ identifies the neighbors of a vertex """
        neighs = []
        for i, j in self.adjlist:
            if i == x:
                neighs.append(j)
            elif j == x:
                neighs.append(i)
        return neighs

    def heuristic (self):
        degrees = np.zeros(self.nvertices)
        candidates = list (range(self.nvertices))

        sol = []

        for x, y in self.adjlist:
            degrees[x] += 1
            degrees[y] += 1

        while (candidates != []):
            _, candidate = min([(degrees[i] ,i )for i in candidates])
            candidates.remove(candidate)
            sol.append(candidate)

            neighbors = self.neighbors(candidate)
            for neigh in neighbors:
                try:
                    candidates.remove(neigh) #delete the neighbors
                except:
                    pass
                for neighInd in self.neighbors(neigh):
                    degrees[neighInd] -= 1 #decreases the degree of the indirect neighbors 
        
        opt = sum(self.weights[i] for i in sol)
        return opt, sol
                




def generateIS (nvertices, maxweight, density):
    adjlist = []
    weights = []

    for i in range (nvertices):
        weights.append (random.randint(1,maxweight))
        for j in range (i, nvertices):
            rnd =  random.uniform(0,1)
            if (rnd <= density):
                adjlist.append([i,j])
    return IndependentSet (nvertices, weights, adjlist)
        
                
