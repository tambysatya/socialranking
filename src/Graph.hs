{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Graph where

import qualified Data.IntMap as M
import qualified Data.IntSet as S
import Control.Lens
import Data.Maybe
import System.Random
import System.Random.Shuffle
import Control.Monad
import Data.Foldable
import Data.Function
import qualified Data.List as L


import Types

newtype Weight = Weight {_fromWeight :: Int}
    deriving (Num,Eq,Ord)
type Graph = M.IntMap (Weight,M.IntMap Weight)


neighbors :: Graph -> Int -> [Int]
neighbors g v = M.keys $ snd $ fromJust $ g ^. at v

degree :: Graph -> Int -> Int
degree g v = length $ neighbors g v

weight :: Graph -> Int -> Weight
weight g v = fst $ fromJust $ g ^. at v

edgeList :: Graph -> [((Int,Int),Weight)]
edgeList g = [((vi,vj),w) | vi <- vs, (vj,w) <- M.assocs $ snd $ fromJust $ g ^. at vi, vj > vi]
    where vs = M.keys g


insertDirectedEdgesV :: Graph -> Int -> [(Int, Weight)] -> Graph
insertDirectedEdgesV g i js = case i `M.lookup` g of
                                    Nothing -> error $ "node " ++ show i ++ " does not exists"
                                    Just (val,neighs) -> g & at i .~ Just (val, M.union (M.fromList js) neighs)
insertUndirectedEdgesV :: Graph -> Int -> [(Int,Weight)] -> Graph
insertUndirectedEdgesV gr src dsts = foldr f (insertDirectedEdgesV gr src dsts) dsts
    where f (el,w) acc = insertDirectedEdgesV acc el [(src,w)]

insertDirectedEdges g i js = insertDirectedEdgesV g i $ zip js (repeat 0)
insertUndirectedEdges g i js = insertUndirectedEdgesV g i $ zip js (repeat 0)

createNodeV :: Weight -> Graph -> Int -> IO Graph
createNodeV (Weight maxw) gr i = do
    v <- randomRIO (1,maxw)
    pure $ gr & at i .~ Just (Weight v,M.empty)
createNode :: Graph -> Int -> IO Graph
createNode gr i = createNodeV (Weight 0) gr i
    


{-| Remove a nodes from the graph and from the neighborhood lists -}
deleteNode :: Graph -> Int -> Graph
deleteNode gr i = gr' & at i .~ Nothing 
    where deleteNeighbor :: Graph -> Int -> Graph
          deleteNeighbor gr j = gr & at j ._Just. _2 %~ M.delete i
          gr' :: Graph
          gr' = foldl' deleteNeighbor gr (neighbors gr i) -- the neighbors forget i

{-| Computes the subgraph restricted to a subset of nodes -}
subGraph :: Graph -> [Int] -> Graph
subGraph g ls = foldl' deleteNode g $ vertices L.\\ ls
    where vertices = M.keys g
subGraphFromEdge :: Graph -> [((Int,Int),Weight)] -> Graph
--subGraphFromEdge gr l = subGraph gr allowedVertices
subGraphFromEdge g l = fmap (\(w,neighs) -> (w, M.restrictKeys neighs allowedVertices)) $ M.restrictKeys g allowedVertices
    where edges = fst <$> l
          allowedVertices = S.fromList $ L.nub $ fmap fst edges ++ fmap snd edges
    

mkRandomGraph :: Weight -> Double -> Int -> IO Graph
mkRandomGraph maxweight prob siz = foldM f M.empty [1..siz]
    where f acc el =  do 
                acc' <- createNodeV maxweight acc el

                let sampleEdge ni = do
                        pr <- randomRIO (0,1)
                        pure $ pr <= prob
                samples <- mapM sampleEdge [el-1,el-2 .. 1]
                let nedges = [i | (i,si) <- zip [el-1, el-2..1] samples, si == True]
                pure $ insertUndirectedEdges acc' el nedges

mkRandomGraphW :: Weight -> Double -> Int -> IO Graph
mkRandomGraphW maxweight prob siz = foldM f M.empty [1..siz]
    where f acc el =  do 
                acc' <- createNodeV maxweight acc el

                let sampleEdge ni = do
                        pr <- randomRIO (0,1)
                        pure $ pr <= prob
                samples <- mapM sampleEdge [el-1,el-2 .. 1]
                let nedges = [i | (i,si) <- zip [el-1, el-2..1] samples, si == True]
                weights <- sequence [Weight <$> randomRIO (1,_fromWeight maxweight) | _ <- nedges]
                pure $ insertUndirectedEdgesV acc' el $ zip nedges weights


instance Show Weight where show (Weight v) = "W:" ++ show v

{- Remove a nodes and all of its neighbors -}
isUpdate :: Graph -> Int -> Graph
isUpdate gr i = deleteNode (foldl' deleteNode gr (neighbors gr i)) i

{-independent set trivial solver -}

score :: Graph -> [Int] -> Weight
score gr sol = sum [fst $ fromJust (gr ^. at vi) | vi <- sol]
isSolver :: Graph -> [Int]
isSolver gr 
            | null candidates = []
            | otherwise = let (c:cs) = candidates 
                              without = isSolver (deleteNode gr c) -- only the node is deleted
                              with = c: isSolver (isUpdate gr c)
                          in if score gr without >= score gr with then without else with
        where candidates = M.keys gr
isGreedLen gr = isGreed' gr
    where isGreed' g 
                | M.null g = []
                | otherwise = let (v:vs) = L.sortBy (compare `on` (degree g)) $ M.keys g
                                  g' = foldr (\el acc -> deleteNode acc el) g (neighbors g v)
                              in v:(isGreed' $ deleteNode g' v)
                                  
isGreedMax gr = isGreed' gr
    where isGreed' g 
                | M.null g = []
                | otherwise = let (v:vs) = L.sortBy (compare `on` f) $ M.keys g
                                  g' = foldr (\el acc -> deleteNode acc el) g (neighbors g v)
                                  f vi = - (fromIntegral $ _fromWeight $ weight g vi) / (fromIntegral $ degree g vi + 1)
                              in v:(isGreed' $ deleteNode g' v)
 

