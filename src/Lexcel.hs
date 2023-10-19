module Lexcel where

import MIP.Class

import qualified Data.List as L
import Data.Containers.ListUtils
import System.Random.Shuffle
import Control.Monad

import Data.Function


type Coalition a = [a]


countOccurences :: (Eq a) => [Coalition a] -> a -> Int
countOccurences [] _ = 0
countOccurences (l:ls) x
    | x `elem` l = 1 + countOccurences ls x
    | otherwise = 0 + countOccurences ls x


lexcel_ :: (Eq a) => [a] -> [[Coalition a]] -> [a]
lexcel_ individuals groupedcoals = reverse $ L.sortBy (compare `on` lexcelVector) individuals
    where lexcelVector x = [countOccurences l x | l <- groupedcoals]


lexcel :: (Eq a, Ord a) => Int -> [a] -> [(Double, Coalition a)] -> [a]
lexcel nclasses individuals scores = lexcel_ individuals classes
    where uniq = nubOrdOn snd scores 
          classes = reverse $ chunk chunksize $ map snd $ L.sortBy (compare `on` fst) uniq
          chunksize = truncate $ fromIntegral (length scores) / fromIntegral nclasses


chunk :: Int -> [Coalition a] -> [[Coalition a]]
chunk _ chunksize = []
chunk n l = let (begin, end) = L.splitAt n l
          in begin: chunk n end


randomCoal :: Problem p => p -> IO (Coalition Int)
randomCoal pb = shuffleM $ individuals pb

lexcelGreedy :: Problem p => Int -> Int -> p -> IO (Double, Double)
lexcelGreedy nclasses ncoals pb = do
    coals <- sequence [randomCoal pb | i <- [1..ncoals]] :: IO [Coalition Int]
    let uniq = [ (value pb coal, coal) | coal <-  L.nub coals] 
        maxval  = L.maximum $ fst <$> uniq -- bestcoalition 
    putStrLn $ show (length uniq) ++ " solutions."
    pure $ (value pb $ lexcel nclasses (individuals pb) uniq, maxval)







