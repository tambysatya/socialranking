{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Graph where

import qualified Data.IntMap as M
import Control.Lens
import Data.Maybe
import System.Random
import Control.Monad
import Data.Foldable

newtype Weight = Weight Int
    deriving (Num,Eq,Ord)
type Graph = M.IntMap (Weight,M.IntMap ())


neighbors :: Graph -> Int -> [Int]
neighbors g v = M.keys $ snd $ fromJust $ g ^. at v


insertDirectedEdges :: Graph -> Int -> [Int] -> Graph
insertDirectedEdges g i js = case i `M.lookup` g of
                                    Nothing -> error $ "node " ++ show i ++ " does not exists"
                                    Just (val,neighs) -> g & at i .~ Just (val, M.union (M.fromList $ zip js $ repeat ()) neighs)
insertUndirectedEdges :: Graph -> Int -> [Int] -> Graph
insertUndirectedEdges gr src dsts = foldr f (insertDirectedEdges gr src dsts) dsts
    where f el acc = insertDirectedEdges acc el [src]

createNode :: Weight -> Graph -> Int -> IO Graph
createNode (Weight maxw) gr i = do
    v <- randomRIO (1,maxw)
    pure $ gr & at i .~ Just (Weight v,M.empty)


deleteNode :: Graph -> Int -> Graph
deleteNode gr i = gr' & at i .~ Nothing 
    where deleteNeighbor :: Graph -> Int -> Graph
          deleteNeighbor gr j = gr & at j ._Just. _2 %~ M.delete i
          gr' :: Graph
          gr' = foldl' deleteNeighbor gr (neighbors gr i) -- the neighbors forget i

mkRandomGraph :: Weight -> Double -> Int -> IO Graph
mkRandomGraph maxweight prob siz = foldM f M.empty [1..siz]
    where f acc el =  do 
                acc' <- createNode maxweight acc el

                let sampleEdge ni = do
                        pr <- randomRIO (0,1)
                        -- debug
                        --print (ni,pr)
                        when (pr <= prob) $ putStrLn $ "adding: " ++ show (el,ni)

                        pure $ pr <= prob
                samples <- mapM sampleEdge [el-1,el-2 .. 1]
                let nedges = [i | (i,si) <- zip [el-1, el-2..1] samples, si == True]
                pure $ insertUndirectedEdges acc' el nedges

instance Show Weight where show (Weight v) = "W:" ++ show v


{-independent set trivial solver -}
{-
isSolver :: Graph -> [Int] -> (Weight,[Int])
isSolver gr [] = (0,[])
isSolver gr (c:cs) = maxFst (isSolver gr cs) (isSolver deleteEdgeC cs )-- [ci | ci <- cand]
    where score :: [Int] -> Weight
          score sol = sum [fst $ fromJust (gr ^. at vi) | vi <- sol]
          maxFst (x,v) (x',v')
                        | v >= v' = (x,v)
                        | else = (x',v')
  -}        
