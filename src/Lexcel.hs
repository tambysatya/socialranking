module Lexcel where


import qualified Data.List as L
import Data.Function


type Coalition a = [a]


countOccurences :: (Eq a) => [Coalition a] -> a -> Int
countOccurences [] _ = 0
countOccurences (l:ls) x
    | x `elem` l = 1 + countOccurences ls x
    | otherwise = 0 + countOccurences ls x


lexcel :: (Eq a) => [a] -> [[Coalition a]] -> [a]
lexcel individuals groupedcoals = reverse $ L.sortBy (compare `on` lexcelVector) individuals
    where lexcelVector x = [countOccurences l x | l <- groupedcoals]
