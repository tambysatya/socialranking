{-# LANGUAGE BangPatterns #-}
module Lexcel where

import MIP.Class

import MIP.KP

import qualified Data.List as L
import qualified Data.IntSet as I
import qualified Data.Set as S
import Data.Containers.ListUtils
import System.Random.Shuffle
import System.Random
import Control.Monad

import IPSolver (IloEnv)

import Data.Function
import Debug.Trace


type Coalition= I.IntSet


countOccurences :: [Coalition] -> Int -> Int
countOccurences [] _ = 0
countOccurences (l:ls) x
    | x `I.member` l = 1 + countOccurences ls x
    | otherwise = 0 + countOccurences ls x


lexcel_ :: [Int] -> [[Coalition]] -> [Int]
lexcel_ individuals groupedcoals = reverse $ L.sortBy (compare `on` lexcelVector) individuals
    where lexcelVector x = [countOccurences l x | l <- groupedcoals]


lexcel :: Int -> [Int] -> [(Double, Coalition)] -> [Int]
lexcel nclasses individuals scores = lexcel_ individuals classes
    where uniq = nubOrdOn snd scores 
          classes = reverse $ chunk chunksize $ map snd $ L.sortBy (compare `on` fst) uniq
          chunksize = truncate $ fromIntegral (length scores) / fromIntegral nclasses


chunk :: Int -> [a] -> [[a]]
chunk _ [] = []
chunk n l = let (begin, end) = L.splitAt n l
          in begin: chunk n end


randomOrder :: Problem p => p -> IO [Int]
randomOrder pb = shuffleM $ individuals pb

randomOrder' :: Problem p => p -> Int -> IO [Int] -- with fixed size
randomOrder' pb siz = take siz <$> randomOrder pb

biasedOrder :: Problem p => p -> S.Set (Double, Int) -> IO [Int]
biasedOrder pb !ws 
    | S.empty == ws = pure []
    | otherwise = do
        sample <- sampleWithWeight $ S.toList ws
        let ws' = S.delete sample ws
        (snd sample:) <$> biasedOrder pb ws'

sampleWithWeight :: [(Double,a)] -> IO (Double,a)
sampleWithWeight ws = do
        rnd <- randomRIO (0, total)
        pure $ choose ws 0 rnd
    where total = sum $ fmap fst ws
          choose [] _ _ = error "invalid sample"
          choose ((w,x):xs) begin rnd
                    | rnd >= begin && rnd < begin+w = (w,x)
                    | otherwise = choose xs (begin+w) rnd

lexcelGreedy :: Problem p => Int -> Int -> Maybe Int -> p -> IO (Double, Double)
lexcelGreedy nclasses ncoals coalsizeM pb = do
    orders <- case coalsizeM of 
                    Just l -> sequence [take l <$> randomOrder pb | i <- [1..ncoals]] :: IO [[Int]]
                    Nothing -> sequence [randomOrder pb | i <- [1..ncoals]] :: IO [[Int]]
    let uniq = [ value pb coal | coal <-  nubOrd orders] 
        maxval  = L.maximum $ fst <$> uniq -- bestcoalition 
    putStrLn $ show (length uniq) ++ " solutions."
    pure $ (fst $ value pb $ lexcel nclasses (individuals pb) uniq, maxval)

biasedLexcelGreedy :: Problem p => Int -> Int -> Maybe Int -> p -> IO (Double,Double)
biasedLexcelGreedy nclasses ncoals coalsizeM pb = do
    orders <- case coalsizeM of 
                    Just l -> sequence [take l <$> biasedOrder pb ws | i <- [1..ncoals]] :: IO [[Int]]
                    Nothing -> sequence [biasedOrder pb ws | i <- [1..ncoals]] :: IO [[Int]]
    let uniq = [ value pb coal | coal <-  nubOrd orders] 
        maxval  = L.maximum $ fst <$> uniq -- bestcoalition 
    putStrLn $ show (length uniq) ++ " solutions."
    pure $ (fst $ value pb $ lexcel nclasses (individuals pb) uniq, maxval)
  where effs = efficiency pb <$> individuals pb
        ws = S.fromList $ zip effs $ individuals pb



lexcelKP :: IloEnv -> Int -> Int -> Int -> KPIns -> IO (Double, Double)
lexcelKP env nclasses ncoals coalsiz kp = do
    orders <- sequence [randomOrder' kp coalsiz | i <- [1..ncoals]] :: IO [[Int]]
    lp <- buildKP env kp

    scores <- forM orders $ \o -> MIP.KP.solveCoalition (kp, lp) $ I.fromList o

    let maxval = L.maximum $ fst <$> scores
        uniq  = nubOrdOn snd scores
    putStrLn $ show (length uniq) ++ " solutions."
    pure $ (fst $ value kp $ lexcel nclasses (individuals kp) uniq, maxval)





