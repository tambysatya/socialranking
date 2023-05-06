
import torch
import cplex
import itertools

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
        
        

    
