module Main where

import Utils
import Test
import Lexcel
import qualified Data.List as L
import qualified Data.IntSet as I
import MIP.Class
import MIP.KP as KP
import MIP.IS as IS
import IPSolver (IloEnv, newIloEnv)
import qualified Data.Set as S
import Control.Monad




--main = mainKP 100 50
--main = mainKP 500 50
main = mainIS 100  50



mainKP nitems nctrs  = do
    forM_ nclasses $ \nclass ->
        forM_ ncoals $ \ncoal ->
            forM_ coalsizes $ \coalsize ->
                testKP nitems nctrs 50 nclass ncoal (Just coalsize) 20
  where nclasses = [10, 50]
        ncoals = [500,1000]
        coalsizes = [3,5,10]
    


{-
mainKP200_50 = do
    --     items ctrs wmax nclasses ncoals coalsizeM ntests
    testKP   200   50   50       10   1000  (Just  5)    20
    testKP   200   50   50       10   1000  (Just 10)    20
    testKP   200   50   50       10   1000  (Just 15)    20
    testKP   200   50   50       10   2000  (Just  5)    20
    testKP   200   50   50       10   2000  (Just 10)    20
    testKP   200   50   50       10   2000  (Just 15)    20
    testKP   200   50   50       20   1000  (Just  5)    20
    testKP   200   50   50       20   1000  (Just 10)    20
    testKP   200   50   50       20   1000  (Just 15)    20
    testKP   200   50   50       20   2000  (Just  5)    20
    testKP   200   50   50       20   2000  (Just 10)    20
    testKP   200   50   50       20   2000  (Just 15)    20
    testKP   200   50   50       20   1000  (Just  5)    20
    testKP   200   50   50       20   1000  (Just 10)    20
    testKP   200   50   50       20   1000  (Just 15)    20



mainKP100_50 = do
    --     items ctrs wmax nclasses ncoals coalsizeM ntests
    testKP   100   50   50       10   1000  (Just  5)    20
    testKP   100   50   50       10   1000  (Just 10)    20
    testKP   100   50   50       10   1000  (Just 15)    20
    testKP   100   50   50       10   2000  (Just  5)    20
    testKP   100   50   50       10   2000  (Just 10)    20
    testKP   100   50   50       10   2000  (Just 15)    20
    testKP   100   50   50       20   1000  (Just  5)    20
    testKP   100   50   50       20   1000  (Just 10)    20
    testKP   100   50   50       20   1000  (Just 15)    20
    testKP   100   50   50       20   2000  (Just  5)    20
    testKP   100   50   50       20   2000  (Just 10)    20
    testKP   100   50   50       20   2000  (Just 15)    20
    testKP   100   50   50       20   1000  (Just  5)    20
    testKP   100   50   50       20   1000  (Just 10)    20
    testKP   100   50   50       20   1000  (Just 15)    20

-}
mainIS nvertices wmax = do
    forM_ densities $ \density ->
        forM_ nclasses $ \nclass ->
            forM_ ncoals $ \ncoal -> 
                forM_ csizes $ \csize ->
                    testIS nvertices wmax density nclass ncoal (Just csize) 20
 where densities = [0.1,0.5,0.7]
       nclasses = [10,50]
       ncoals = [500,1000]
       csizes = [3,5,10]


mainIS' = do
    --     nvertices wmax density nclasses ncoals coalsizeM ntests
    testIS       100   50     0.1       10   1000  (Just 5)     20
    testIS       100   50     0.1       10   1000 (Just 10)     20
    testIS       100   50     0.1       10   1000 (Just 15)     20
    testIS       100   50     0.1       10   2000  (Just 5)     20
    testIS       100   50     0.1       10   2000 (Just 10)     20
    testIS       100   50     0.1       10   2000 (Just 15)     20
    testIS       100   50     0.1       20   1000  (Just 5)     20
    testIS       100   50     0.1       20   1000 (Just 10)     20
    testIS       100   50     0.1       20   1000 (Just 15)     20
    testIS       100   50     0.1       20   2000  (Just 5)     20
    testIS       100   50     0.1       20   2000 (Just 10)     20
    testIS       100   50     0.1       20   2000 (Just 15)     20
    testIS       100   50     0.5       10   1000  (Just 5)     20
    testIS       100   50     0.5       10   1000 (Just 10)     20
    testIS       100   50     0.5       10   1000 (Just 15)     20
    testIS       100   50     0.5       10   2000  (Just 5)     20
    testIS       100   50     0.5       10   2000 (Just 10)     20
    testIS       100   50     0.5       10   2000 (Just 15)     20
    testIS       100   50     0.5       20   1000  (Just 5)     20
    testIS       100   50     0.5       20   1000 (Just 10)     20
    testIS       100   50     0.5       20   1000 (Just 15)     20
    testIS       100   50     0.5       20   2000  (Just 5)     20
    testIS       100   50     0.5       20   2000 (Just 10)     20
    testIS       100   50     0.5       20   2000 (Just 15)     20
    testIS       100   50     0.7       10   1000  (Just 5)     20
    testIS       100   50     0.7       10   1000 (Just 10)     20
    testIS       100   50     0.7       10   1000 (Just 15)     20
    testIS       100   50     0.7       10   2000  (Just 5)     20
    testIS       100   50     0.7       10   2000 (Just 10)     20
    testIS       100   50     0.7       10   2000 (Just 15)     20
    testIS       100   50     0.7       20   1000  (Just 5)     20
    testIS       100   50     0.7       20   1000 (Just 10)     20
    testIS       100   50     0.7       20   1000 (Just 15)     20
    testIS       100   50     0.7       20   2000  (Just 5)     20
    testIS       100   50     0.7       20   2000 (Just 10)     20
    testIS       100   50     0.7       20   2000 (Just 15)     20





--mainKP = testKP 100 50 50 20 1000 (Just 5) 10
mainCpxKP = cpxKP 100 50 50 10 1000 10 10


cpxKP :: Int -> Int -> Int -> Int -> Int -> Int -> Int -> IO ()
cpxKP nitems nctrs maxval nclasses ncoals coalsiz ntests = do
    env <- newIloEnv
    results <- forM [1..ntests] $ \_ -> do
        pb <- generateUniformFeasibleKP nitems nctrs maxval
        testKPInstance env nclasses ncoals coalsiz pb
    let (opts,lexs,hs,maxs) = L.unzip4 results
    putStrLn "---"
    putStrLn $ "opt=" ++ show (avg opts)
    putStrLn $ "lex=" ++ show (avg lexs)
    putStrLn $ "greedy=" ++ show (avg hs)
    putStrLn $ "maxs=" ++ show (avg maxs)
    pure ()




-- tests with cplex
testKPInstance :: IloEnv -> Int -> Int -> Int -> KPIns -> IO (Double,Double,Double,Double) 
testKPInstance env nclasses ncoals coalsiz pb = do
    (lex, maxval) <- lexcelKP env nclasses ncoals coalsiz pb
    opt <- solve env pb
    let lexratio = lex/opt
        maxratio = maxval/opt
        hratio = heuristic/opt
    
    putStrLn $ "opt="++show opt ++ " lex=" ++ show lex  ++ " greedy=" ++ show heuristic ++ " max=" ++ show maxval
    pure (opt,lexratio,hratio,maxratio)
  where heuristic = greedy pb


