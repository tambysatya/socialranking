module Main where

import Lexcel
import qualified Data.List as L
import MIP.Class
import MIP.KP as KP
import MIP.IS as IS
import IPSolver (IloEnv, newIloEnv)
import qualified Data.Set as S
import Control.Monad


main = mainKP

mainIS = testIS 100 50 0.1 100 1000 10
mainKP = testKP 100 50 50 1000 10000 10


testKP :: Int -> Int -> Int -> Int -> Int -> Int -> IO ()
testKP nitems nctrs maxval nclasses ncoals ntests = do
    env <- newIloEnv
    results <- forM [1..ntests] $ \_ -> do
        pb <- generateUniformFeasibleKP nitems nctrs maxval
        testInstance env nclasses ncoals pb
    let (opts,lexs,hs,maxs) = L.unzip4 results
    putStrLn "---"
    putStrLn $ "opt=" ++ show (avg opts)
    putStrLn $ "lex=" ++ show (avg lexs)
    putStrLn $ "greedy=" ++ show (avg hs)
    putStrLn $ "maxs=" ++ show (avg maxs)
    pure ()

testIS :: Int -> Int -> Double -> Int -> Int -> Int -> IO ()
testIS nvertices maxval density nclasses ncoals ntests = do
    env <- newIloEnv
    results <- forM [1..ntests] $ \_ -> do
        pb <- generateIS nvertices maxval density
        testInstance env nclasses ncoals pb
    let (opts,lexs,hs,maxs) = L.unzip4 results
    putStrLn "---"
    putStrLn $ "opt=" ++ show (avg opts)
    putStrLn $ "lex=" ++ show (avg lexs)
    putStrLn $ "greedy=" ++ show (avg hs)
    putStrLn $ "maxs=" ++ show (avg maxs)
    pure ()
testInstance :: Problem p => IloEnv -> Int -> Int -> p -> IO (Double,Double,Double,Double)
testInstance env nclasses ncoals pb = do
    (lex, maxval) <- lexcelGreedy nclasses ncoals pb
    opt <- solve env pb
    let lexratio = lex/opt
        maxratio = maxval/opt
        hratio = heuristic/opt
    
    putStrLn $ "opt="++show opt ++ " lex=" ++ show lex  ++ " greedy=" ++ show heuristic ++ " max=" ++ show maxval
    pure (opt,lexratio,hratio,maxratio)
  where heuristic = greedy pb

avg l = sum l / fromIntegral (length l)
