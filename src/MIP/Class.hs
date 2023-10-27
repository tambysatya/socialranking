module MIP.Class where


import IPSolver
import qualified Data.IntSet as I

class Problem p where
    individuals :: p -> [Int]
    value :: p -> [Int] -> (Double, I.IntSet) 
    greedy :: p -> Double
    solve :: IloEnv -> p -> IO Double
    efficiency :: p -> Int -> Double


