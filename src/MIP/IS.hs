module MIP.IS where

import IPSolver
import qualified Data.Array as A
import qualified Data.Set as S
import qualified Data.List as L
import Utils
import Control.Monad
import System.Random
import Control.Lens
import Data.Function

data ISIns = ISIns { _adjlist :: [(Int,Int)], _weights :: A.Array Int Int}
data ISCpx = ISCpx {_isMdl :: IloModel, _isCpx :: IloCplex, _isObj :: IloObjective, _isCtrs :: A.Array Int IloRange, _isVars :: A.Array Int IloBoolVar}


adjacents xi edges = [(x,y) | (x,y) <- edges , xi == x || xi == y ] 


buildIS :: IloEnv -> ISIns -> IO ISCpx
buildIS env is = do
    mdl <- newIloObject env
    cpx <- newIloObject env
    cpx `extract` mdl
    
    obj <- newIloObject env
    setMaximize obj >> mdl `add` obj

    vars <- mkVector <$> sequence [newIloObject env | _ <- [1..nvars]] :: IO (A.Array Int IloBoolVar)

    forM_ (zip (A.elems vars) $ (A.elems $ _weights is)) $ \(xi, wi) -> setLinearCoef obj xi $ fromIntegral wi
    ctrs <- forM edges $ \(xi, xj) -> do
        ctr <- newIloObject env
        setLinearCoef ctr (vars A.! xi) 1 >> setLinearCoef ctr (vars A.! xj) 1
        setUB ctr 1
        mdl `add` ctr
        pure ctr
    pure $ ISCpx mdl cpx obj (mkVector ctrs) vars


 where 
       edges = _adjlist is
       (_,nvars) = A.bounds $ _weights is



generateIS :: Int -> Int -> Double -> IO ISIns
generateIS nvertices maxval density = do
    ws <- sequence [randomRIO (1, maxval) | _ <- [1..nvertices]]
    edges <- forM [1..nvertices] $ \i ->
             forM [i..nvertices] $ \j -> do
                r <- randomRIO (0,1)
                if r <= density then pure ((i,j),True) else pure ((i,j),False)
    let adjlist = [e | (e, b) <- concat edges, b == True]                
    pure $ ISIns adjlist $ mkVector ws


solveCoalition :: (ISIns, ISCpx) -> S.Set Int -> IO (Int, [Int])
solveCoalition (ins, pb) coal = do
    forM_ todelete $ \i ->
        setLinearCoef obj (vars A.! i) 0
    solve (_isCpx pb)    
    opt <- getObjValue (_isCpx pb)
    sol <- sequence $ fmap (_isCpx pb `getValue`) vars

    forM_ todelete $ \i ->
        setLinearCoef obj (vars A.! i) (fromIntegral $ weights A.! i)

    pure (round opt, [i | (i,si) <- zip [1..] $ A.elems sol, si == 1, i `S.member` coal])
    
 where  nvars = length (_weights ins)
        todelete = S.toList $ S.fromList [1..nvars] S.\\ coal
        vars = _isVars pb
        obj = _isObj pb
        weights = _weights ins



greedy :: ISIns -> [Int] -> Double
greedy ins order = greedy' (_weights ins) (S.fromList $ _adjlist ins) order $ S.fromList order
greedy' :: A.Array Int Int -> S.Set (Int,Int) -> [Int] -> S.Set Int -> Double
greedy' _ _ [] _ = 0
greedy' weights adjlist (x:xs) candidates
    | S.empty == candidates = 0
    | not $ x `S.member` candidates = greedy' weights adjlist xs candidates
    | otherwise = fromIntegral (weights A.! x) + greedy' weights adjlist' xs candidates'
  where adjedges = adjacents x $ S.toList adjlist -- adjacent edges
        neighbors = S.fromList $ L.nub $ map fst adjedges ++ map snd adjedges
        adjlist' = adjlist S.\\ S.fromList adjedges
        candidates' = candidates S.\\ neighbors
        


heuristic ins = heuristic' neighbors degrees (_weights ins) candidates
    where neighbors = foldr (\(xi,xj) acc -> acc & ix xi %~ (xj:)
                                                 & ix xj %~ (xi:)) (mkVector [[] | _ <- [1..nvertices]]) (_adjlist ins)
          degrees = fmap length neighbors
          nvertices = length $ _weights ins
          candidates = S.fromList [1..nvertices]

heuristic' :: A.Array Int [Int] -> A.Array Int Int -> A.Array Int Int -> S.Set Int -> Int
heuristic' neighbors degrees weights candidates 
        | S.empty == candidates = 0
        | otherwise = weights A.! best + heuristic' neighbors degrees' weights (candidates S.\\ S.fromList (best:neighbors A.! best)) 
    where efficiency i = (fromIntegral $ weights A.! i) / (1e-9 + fromIntegral (degrees A.! i))
          best = L.maximumBy (compare `on` efficiency) $ S.toList candidates
          removeNeigh :: Int -> A.Array Int Int -> A.Array Int Int
          removeNeigh ni degs = foldr (\el acc -> acc & ix el %~ (subtract 1)) degs (neighbors A.! ni)

          degrees' = foldr removeNeigh degrees (neighbors A.! best)


