module Main where

import Lexcel
import qualified Data.List as L
import MIP.KP
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
    kp <- generateUniformFeasibleKP 100 50 100
    env <- newIloEnv
    pb <- buildKP env kp
    ret <- solveCoalition (kp, pb) $ S.fromList [1..100]
    print ret
    print $ heuristic kp
