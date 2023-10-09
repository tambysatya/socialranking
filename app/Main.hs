module Main where

import Lexcel
import qualified Data.List as L
import MIP.KP as KP
import MIP.IS as IS
import IPSolver
import qualified Data.Set as S


class1 = [[2,3], [2,5], [3,5], [2,9]]
class2 = [[2,1], [3,5],[3,9]]
class3 = [[2,5], [2,9]]

coals :: [[[Int]]]
coals = [class1,class2,class3]

individuals :: (Eq a) => [[Coalition a]] -> [a]
individuals groupedcoals = L.nub $ [individual | groups <- groupedcoals, coal <- groups, individual <- coal]

main = do
    is <- generateIS 50 100 0.2
    env <- newIloEnv
    pb <- buildIS env is
    ret <- IS.solveCoalition (is,pb) $ S.fromList [1..50]
    print ret
    print $ IS.heuristic is
    pure ()
main' = do
    kp <- generateUniformFeasibleKP 100 50 100
    env <- newIloEnv
    pb <- buildKP env kp
    ret <- KP.solveCoalition (kp, pb) $ S.fromList [1..100]
    print ret
    print $ KP.heuristic kp
