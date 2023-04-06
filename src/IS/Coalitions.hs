module IS.Coalitions where

import Graph
import Types 

import qualified Data.Set as S
import qualified Data.IntMap as M
import qualified Data.List as L
import Data.Function
import System.Random.Shuffle
import System.Random
import Control.Monad


{-
atMostNOrder :: Graph 
             -> Int -- number of iterations
             -> Int -- maximum length
             -> IO (TotalPreorder (Coalition Int))
atMostNOrder gr nit maxlen = do
        coals <- genAtMostN gr nit maxlen
        mkOrderIO (\l -> subGraph gr l) (toList <$> S.toList coals)
-}
genCoalition :: Graph -> Int -> IO (Coalition Int)
genCoalition gr siz = mkCoalition . take siz <$> shuffleM [1..n]
    where n = M.size gr


genAtMostN :: Graph
           -> Int  -- number of iterations
           -> Int  -- maximum length
           -> IO (S.Set (Coalition Int)) --
genAtMostN gr nIt maxlen = genAtMostN' gr nIt maxlen S.empty
genAtMostN' :: Graph -> Int -> Int -> S.Set (Coalition Int) -> IO (S.Set (Coalition Int))
genAtMostN' _ 0 _ ret = pure ret
genAtMostN' gr nIt maxlen ret = do
            siz <- randomRIO (1, maxlen)
            coal <- genCoalition gr siz
            genAtMostN' gr (nIt-1) maxlen (S.insert coal ret)
            


mkOrderIO :: (Ord a) => (Coalition a -> IO Int) -> [Coalition a] -> IO (TotalPreorder (Coalition a))
mkOrderIO scorefun coals =  do 
    scores <- sequence $ scorefun <$> coals
    pure $ mkOrder $ zip scores coals
