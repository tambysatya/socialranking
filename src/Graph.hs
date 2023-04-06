{-# LANGUAGE GeneralizedNewtypeDeriving #-}
module Graph where

import qualified Data.IntMap as M
import Control.Lens
import Data.Maybe
import System.Random
import System.Random.Shuffle
import Control.Monad
import Data.Foldable
import qualified Data.List as L


import Types

newtype Weight = Weight {_fromWeight :: Int}
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

-- Tests

order gr = mkCoalitionOrder (_fromWeight . score gr) (isSolver . subGraph gr <$> allSubsets [1..n])
    where n = M.size gr

glouton :: Graph -> [Int] -> [Int]
glouton gr [] = []
glouton gr (c:candidates) = c:glouton gr (candidates L.\\ neighbors gr c)

randomSol :: Graph -> IO [Int]
randomSol gr = glouton gr <$> shuffleM (M.keys gr)

test :: Double -> Int -> IO ()
test prob siz = do
    gr <- mkRandomGraph 10 prob siz
    let opt = isSolver gr
        ordret = reverse $ lexCell [1..siz] $ order gr
        ord = fmap snd ordret
        glout = glouton gr ord
    print (score gr opt, score gr glout)
    putStrLn $ "opt=" ++ show opt ++ " lexcell=" ++ show glout ++  " order=" ++ show ord
    putStrLn $ "total order=" ++ unlines (fmap show  ordret)
    rand <- randomSol gr
    putStrLn $ "random=" ++ show rand ++ " val= " ++ show (score gr rand)
