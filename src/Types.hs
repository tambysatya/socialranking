module Types where

import qualified Data.IntMap as M
import qualified Data.Set as S
import qualified Data.List as L
import Data.Function
import Data.Maybe

newtype Coalition a = Coalition (S.Set a)

newtype TotalPreorder a = TotalPreorder (M.IntMap [a])


type Score a = a -> Int


mkCoalition :: (Ord a) => [a] -> Coalition a
mkCoalition l = Coalition $ S.fromList l

toList :: Coalition a -> [a]
toList (Coalition l) = S.toList l

isIn :: (Ord a) => a -> Coalition a -> Bool
isIn el (Coalition s) = el `S.member` s

instance (Ord a) => Semigroup (Coalition a) where
    (Coalition l) <> (Coalition l') = Coalition $ S.union l l'
instance (Ord a) => Monoid (Coalition a) where
    mempty = Coalition S.empty

instance (Semigroup a) => Semigroup (TotalPreorder a) where
    (TotalPreorder l) <> (TotalPreorder l') = TotalPreorder $ M.unionWith (<>) l l'
instance (Monoid a) => Monoid (TotalPreorder a) where
    mempty = TotalPreorder (M.empty)



mkCoalitionOrder :: (Ord a) => Score [a] -> [[a]] -> TotalPreorder (Coalition a)
mkCoalitionOrder score l = TotalPreorder $ M.fromList $ result
    where evaledCoal = L.groupBy ((==) `on` fst) $ L.sortBy (compare `on` fst) [(score li, mkCoalition li) | li <- l]
          result = fmap f  evaledCoal
          f :: [(a,b)] -> (a,[b])
          f l@((s,li):ls) = (s, fmap snd l)


         
instance Show a => Show (Coalition a) where show (Coalition s) = show $ S.toList s
instance Show a => Show (TotalPreorder a) where show (TotalPreorder l) = show $ M.toList l

testPreorder :: TotalPreorder (Coalition Int)
testPreorder = mkCoalitionOrder score coals
    where score = sum
          coals = [[1,2,3],[2,5], [2,3], [1,2], [1], [5]]

allSubsets :: [a] -> [[a]]
--allSubsets [] = []
allSubsets [] = error "allSubsets called on an empty list and cannot contain the empty coalition"
allSubsets [x] = [[x]] 
allSubsets (x:xs) = [[x]] ++ allSubsets xs ++ fmap (x:) (allSubsets xs)

lexCell :: (Ord a) => [a] -> TotalPreorder (Coalition a) -> [([Int],a)]
lexCell eltorank (TotalPreorder l) = ret
    where eqclasses = L.sort $ M.keys l 
          nclasses = M.size l
          lexRank a = [length [ci | ci <- coals, a `isIn` ci]| (i, ri) <- zip [1..] eqclasses , let coals = fromJust $ M.lookup ri l]
          ret = L.sortBy (compare `on` fst) [(lexRank a, a) | a <- eltorank]

printCell :: (Show a) => [([Int],a)] -> IO ()
printCell l = putStrLn $ unlines $ fmap show l


