{-# LANGUAGE TemplateHaskell #-}
module MIP.KP where

import MIP.Class

import Control.Lens
import IPSolver
import qualified Data.Array as A 
import Control.Monad
import System.Random
import qualified Data.Set as S
import qualified Data.IntSet as I
import qualified Data.List as L
import Data.Function
import Utils



data KPIns = KPIns {_profits :: A.Array Int Double, _weights :: A.Array (Int,Int) Double, _capacities :: A.Array Int Double}
data KPCpx = KPCpx {_kpMdl :: !IloModel, _kpCpx :: !IloCplex, _kpObj :: IloObjective, _kpCtrs :: A.Array Int IloRange, _kpVars :: A.Array Int IloBoolVar}

profits :: Lens' KPIns (A.Array Int Double)
profits = lens _profits $ \ins pr -> ins{_profits=pr}

weights:: Lens' KPIns (A.Array (Int,Int) Double)
weights = lens _weights $ \ins w -> ins {_weights=w}

capacities :: Lens' KPIns (A.Array Int Double)
capacities = lens _capacities $ \ins cs -> ins {_capacities = cs}

buildKP :: IloEnv -> KPIns -> IO KPCpx
buildKP env kp = do
    mdl <- newIloObject env
    cpx <- newIloObject env
    cpx `extract` mdl
    
    obj <- newIloObject env
    setMaximize obj >> mdl `add` obj

    vars <- sequence [newIloObject env | vi <- [1..nvars]] :: IO [IloBoolVar]
    
    -- objetive coefficients
    sequence [setLinearCoef obj vi wi | (vi, wi) <- zip vars (A.elems $ _profits kp)]

    -- weights constraints
    let mkConstraint :: IloRange -> Int -> IO ()
        mkConstraint ctr i = sequence_ [setLinearCoef ctr vi wi | (vi, wi) <- zip vars $ getRow (_weights kp) i ]
    ctrs <- forM [1..nctrs] $ \j -> do
        ctr <- newIloObject env
        mkConstraint ctr j 
        ctr `setUB` (_capacities kp A.! j)
        mdl `add` ctr
        pure ctr
    pure $ KPCpx mdl cpx obj (mkVector ctrs) (mkVector vars) 

  where (_,nvars) = A.bounds (_profits kp)
        (_, (nctrs,_)) = A.bounds (_weights kp)


generateUniformFeasibleKP  :: Int -> Int -> Int -> IO KPIns
generateUniformFeasibleKP nitems nctrs maxvals = do
    prs <- sequence [randomRIO (1, maxvals) | i <- [1..nitems]] :: IO [Int]
    weightsmatrix <- forM [1..nctrs] $ \_ -> do
            sequence [randomRIO(1,maxvals) |i <- [1..nitems]]
    let ws = mkArrayFromRows weightsmatrix 
        cmax i = randomRIO (maximum (getRow ws i), sum (getRow ws i))
    
    capacities <- sequence [cmax i | i <- [1..nctrs]]
    pure $ KPIns (fromIntegral <$> mkVector prs) (fromIntegral <$> ws) (fromIntegral <$> mkVector capacities)
    
    


solveCoalition :: (KPIns, KPCpx) -> I.IntSet -> IO (Double, I.IntSet)
solveCoalition (ins, pb) coal = do
        forM todelete $ \i -> 
            setLinearCoef (_kpObj pb) (_kpVars  pb A.! i) 0

        IPSolver.solve (_kpCpx pb)
        opt <- getObjValue (_kpCpx pb)
        sol <- sequence $ fmap (_kpCpx pb `getValue`) (_kpVars pb)

        forM todelete $ \i ->
            setLinearCoef (_kpObj pb) (_kpVars pb A.! i) $ _profits ins A.! i

        pure (fromIntegral $ round opt, I.fromList [i | (i,si) <- zip [1..] $ A.elems sol, si == 1, i `I.member` coal])
    where (_,nitems) = A.bounds (_kpVars pb)
          todelete = I.toList $ I.fromList [1..nitems] I.\\ coal




value_ :: KPIns -> [Int] -> (Double, I.IntSet)
value_ ins [] = (0, I.empty)
value_ ins (x:xs) 
    | and $ fmap (>=0) newcaps = let (opt, sol) = value_ takex xs in (xval+opt, I.insert x sol) -- (_profits ins A.! x) + value_ takex xs
    | otherwise = value_ ins xs
  where takex = ins & capacities .~ mkVector newcaps
        newcaps = [_capacities ins A.! i - _weights ins A.! (i,x) | i <- [1..nctrs] ]
        (_,nctrs) = A.bounds $ _capacities ins
        xval = _profits ins A.! x



heuristic :: KPIns -> Double
heuristic kp = fst $ value_ kp $ reverse $ L.sortBy (compare `on` efficiency) [1..nvars]
    where efficiency x = prs A.! x / sum [ws A.! (i,x) | i <- [1..nctrs]]
          (prs, ws) = (_profits kp, _weights kp)
          (_,(nctrs,nvars)) = A.bounds ws

item_efficiency :: KPIns -> Int -> Double
item_efficiency kp x = prs A.! x / sum [ws A.! (i,x) | i <- [1..nctrs]]
    where (prs, ws) = (_profits kp, _weights kp)
          (_,(nctrs,nvars)) = A.bounds ws



instance Problem KPIns where
    individuals kp  = [n0..n1]
        where (n0,n1) = A.bounds $ _profits kp
    value = value_
    greedy = heuristic
    solve env kp = do
        pb <- buildKP env kp
        let cpx = _kpCpx pb
        IPSolver.solve cpx
        getObjValue cpx
    efficiency = item_efficiency


