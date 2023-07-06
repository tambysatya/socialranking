
import torch
import cplex
import itertools
import sys
import random
from operator import itemgetter

sys.setrecursionlimit(10000)

def init_cplex ():
        '''
            Initialisation de cplex avec des options standard
        '''
        pb = cplex.Cplex()

        #desactivation des logs
        pb.set_log_stream(None)
        pb.set_error_stream(None)
        pb.set_warning_stream(None)
        pb.set_results_stream(None)

        pb.parameters.threads.default =  1

        return pb


class Problem:
    def __init__ (self, objcoefs, A, b): # [Double], [[Double]], [Double]
        """
            objcoefs is the objective function to be MAXIMIZED
            A is the constraint matrix (list of rows)
            b is the right hand side coefs
            all variables are assumed to be binary
        """
        self.var_index = set(range(len(objcoefs)))
        self.pb = init_cplex()

        self.objcoefs = objcoefs
        self.A = A
        self.b = b

        vars = list (map (lambda vi: "x" + str(vi), range (0,len(objcoefs))) )

        types = list( map (lambda vi: 'B', objcoefs))
        self.pb.objective.set_sense(self.pb.objective.sense.maximize)
        self.pb.variables.add(obj=objcoefs, types=types, names=vars)
        #self.pb.variables.add(obj=objcoefs, lb=0, ub=1, types=types, names=vars)
        for row, bi in zip (A,b):
            self.pb.linear_constraints.add (lin_expr=[[vars,row]], senses="L", rhs = [bi] )

    def solve (self):

        self.pb.solve()
        sol = self.pb.solution.get_values()
        opt = self.pb.solution.get_objective_value()
        return (opt, sol)
    
    def solve_coalition(self, coal): # coal :: [Int] which is the indice of variables that can be chosen. All other variables are force to be equal to zero

        """ solves the problem every other variables forced to be equal to zeroes """
        fixed_vars = self.var_index - coal
        self.pb.variables.set_upper_bounds(zip (fixed_vars,itertools.repeat(0)))

        opt, sol = self.solve()
        self.pb.variables.set_upper_bounds(zip (fixed_vars,itertools.repeat(1)))

        return opt, sol

    #def greedy_ (self, order, bs):
    #    if order == []:
    #        return 0
    #    
    #    xi = order.pop(0)
    #    new_bs = bs.clone()

    #    for i, (row, bi) in enumerate(zip (self.A, bs)):
    #        new_bs[i] = bi - row[xi] 
    #        if new_bs[i] < 0: #cannot be added: violates constraint i
    #            #print ("cut=" + xi)
    #            return self.greedy_(order, bs)
    #    ret = self.objcoefs[xi] + self.greedy_(order, new_bs)
    #    return ret
    #
    #def greedy (self, order):
    #    return self.greedy_(order, torch.tensor(self.b))


    def greedy (self, order):
        if order == []:
            return None
        tmpbs = torch.tensor(self.b).clone()
        sol = torch.zeros(len(self.objcoefs))
        coefs = torch.tensor(self.objcoefs)
        opt = 0
        ctrs = torch.tensor(self.A).transpose(1,0)

        for i in order:
            newbs = tmpbs - ctrs[i]
            if (newbs>=0).all():
                sol[i] = 1
                tmpbs = newbs
                opt += coefs[i]
        return opt, sol 
            
    def display (self):
        print ("objs=" + str (self.objcoefs) + "  s.t. " + str(self.A) + " <= " + str(self.b))
    
    def toTensor (self):
        objs = torch.tensor(self.objcoefs)
        A = torch.tensor (self.A)
        b= torch.tensor (self.b)

        return objs, A, b
    def tensorCoalition (self, coal):
        x = torch.zeros(len (self.objcoefs))
        x[torch.tensor(list(coal))] = 1
        return x
    
    def trivial_order (self):
        scores=[]
        for i in range (len(self.objcoefs)):
            w_i = 0
            for j in range (len (self.A)):
                w_i += self.A[j][i]
            scores.append(self.objcoefs[i]/w_i)
        ret = zip (scores, list(range(len(self.objcoefs))))
        ret = sorted(ret, reverse=True)
        return list(map(itemgetter(1),ret))
    def trivial_order_scaled (self):
        scores=[]
        for i in range (len(self.objcoefs)):
            w_i = 0
            for j in range (len (self.A)):
                w_i += self.A[j][i]/self.b[j]
            scores.append(self.objcoefs[i]/w_i)
        ret = zip (scores, list(range(len(self.objcoefs))))
        ret = sorted(ret, reverse=True)
        return list(map(itemgetter(1),ret))



    def toAdjacencyMatrix (self):
        nvars = len(self.objcoefs)
        nctrs = len(self.A)

        #vertices = [vars, ctrs]
        mat = torch.zeros(nvars+nctrs, nvars+nctrs)
        atr = torch.tensor(self.A).transpose(0,1)
        for i in range (nctrs):
            mat[i]= torch.cat((torch.zeros(nvars),atr[i]))/1000

        # making it symetric
        mat_tr = mat.clone().transpose(0,1)
        mat += mat_tr

        #diagonal members
        for i in range (nctrs): #bi
            mat[nvars+i][nvars+i] = self.b[i]/(nvars*1000)
        for i in range (nvars): #cis
            mat[i][i] = self.objcoefs[i]/1000
        return mat
    def density (self, coal):
        score = 0
        for item in coal:
            v, w = 0,0
            v += self.objcoefs[item]
            for weight in range(len(self.A)):
                w += self.A[weight][item]
            score += (v/w)
        return score
    def density_merge (self, coal):
        v, w = 0,0
        for item in coal:
            v += self.objcoefs[item]
            for weight in range(len(self.A)):
                w += self.A[weight][item]
        return v/w
    def density_scaled (self, coal):
        v, w = 0,0
        for item in coal:
            v += self.objcoefs[item]
            for weight in range(len(self.A)):
                w += self.A[weight][item] / self.b[weight]
        return v/w
    

    




            
def random_coalition (individuals, l):
    inds = list(individuals)
    random.shuffle(inds)
    return set(inds[:l])

def random_coalitions (individuals, l, n): 
    """generates at most n coalitions of size n"""
    ret = [random_coalition(individuals,l) for i in range (n)]
    return [k for k, v in itertools.groupby(sorted(ret)) ]

       
        

    
