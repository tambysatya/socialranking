
import torch
import cplex
import itertools
import sys

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

    def greedy_ (self, order, bs):
        if order == []:
            return 0
        
        xi = order.pop(0)
        new_bs = bs.clone()

        for i, (row, bi) in enumerate(zip (self.A, bs)):
            new_bs[i] = bi - row[xi] 
            if new_bs[i] < 0: #cannot be added: violates constraint i
                #print ("cut=" + xi)
                return self.greedy_(order, bs)
        ret = self.objcoefs[xi] + self.greedy_(order, new_bs)
        return ret
    
    def greedy (self, order):
        return self.greedy_(order, torch.tensor(self.b))

    def display (self):
        print ("objs=" + str (self.objcoefs) + "  s.t. " + str(self.A) + " <= " + str(self.b))

            
        
        

    
