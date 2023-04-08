module Utils where

import Graph
import qualified Data.List as L
import System.Random.Shuffle
import qualified Data.IntMap as M

glouton :: Graph -> [Int] -> [Int]
glouton gr [] = []
glouton gr (c:candidates) = c:glouton gr (candidates L.\\ neighbors gr c)

randomSol :: Graph -> IO [Int]
randomSol gr = glouton gr <$> shuffleM (M.keys gr)



data GraphParam = GraphParam {_grsiz :: Int,
                              _wmax :: Weight,
                              _density :: Double
                             }

paramToFile :: String -> GraphParam -> String
paramToFile path (GraphParam siz wmax dens) = path ++ "n-" ++ show siz ++ "_w-" ++ show wmax ++ "_d-" ++ show dens ++ ".dat"

