module MIP.Class where


import IPSolver

class Problem p where
    individuals :: p -> [Int]
    value :: p -> [Int] -> Double 
    greedy :: p -> Double
    solve :: IloEnv -> p -> IO Double


