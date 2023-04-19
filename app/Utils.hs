module Utils (
 module Datas
,glouton, randomSol
)where

import Graph
import Datas
import qualified Data.List as L
import System.Random.Shuffle
import qualified Data.IntMap as M

glouton :: Graph -> [Int] -> [Int]
glouton gr [] = []
glouton gr (c:candidates) = c:glouton gr (candidates L.\\ neighbors gr c)

randomSol :: Graph -> IO [Int]
randomSol gr = glouton gr <$> shuffleM (M.keys gr)




