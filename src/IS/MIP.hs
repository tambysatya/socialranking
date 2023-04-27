module IS.MIP where


import Graph
import IPSolver
import qualified Data.Array as A
import qualified Data.IntMap as M
import Control.Monad
import Control.Lens
import Data.Maybe



testWeightedISMIP = do
    gr <- mkRandomGraph 20 0.5 10
    putStrLn $ unlines [show as | as <- M.assocs gr]
    env <- newIloEnv
    ret <- solveWeightedIS env gr
    putStrLn $ "cpx= " ++ show ret
    putStrLn $ "algo=" ++ show (isSolver gr)

{-| WARNING: vertices must be indiced from one to n -}

solveWeightedIS :: IloEnv -> Graph -> IO (Int,[Int])
solveWeightedIS env gr = case M.keys gr of
                [] -> pure (0,[])
                [x] -> pure (_fromWeight $ score gr [x], [x])
                _ -> do
                    mdl <- newIloObject env
                    cpx <- newIloObject env
                    cpx `extract` mdl

                    obj <- newIloObject env
                    setMaximize obj >> mdl `add` obj

                    vars <- M.fromList . zip vs <$> sequence [newIloObject env | vi <- vs]
                    ctrs <- sequence [mkCtr mdl vars vi | vi <- vs]
                    obj <- sequence [setLinearCoef obj (fromJust $ vars ^. at vi) (fromIntegral $ _fromWeight $ score gr [vi]) | vi <- vs]
                    
                    
                    solve cpx

                    sol <- sequence $ fmap (cpx `getValue`) vars
                    opt <- getObjValue cpx

                    pure (round opt,[i | (i,si) <- M.assocs sol, si == 1])

    where vs = M.keys gr
          mkCtr :: IloModel -> M.IntMap IloBoolVar -> Int -> IO [IloRange]
          mkCtr mdl vars i = forM [ni | ni <- neighbors gr i, ni > i] $ \ni -> do
                                            ct <- newIloObject env
                                            setBounds ct (0,1)
                                            setLinearCoef ct (fromJust $ vars ^. at i) 1
                                            setLinearCoef ct (fromJust $ vars ^. at ni) 1
                                            mdl `add` ct
                                            pure ct
                
solveIS :: IloEnv -> Graph -> IO (Int,[Int])
solveIS env gr = case M.keys gr of
                [] -> pure (0,[])
                [x] -> pure (_fromWeight $ score gr [x], [x])
                _ -> do
                    mdl <- newIloObject env
                    cpx <- newIloObject env
                    cpx `extract` mdl

                    obj <- newIloObject env

                    vars <- M.fromList . zip vs <$> sequence [newIloObject env | vi <- vs]
                    sequence $ fmap (\xi -> setLinearCoef obj xi 1) vars
                    ctrs <- sequence [mkCtr mdl vars vi | vi <- vs]
                    setMaximize obj >> mdl `add` obj
                    
                    solve cpx

                    sol <- sequence $ fmap (cpx `getValue`) vars
                    opt <- getObjValue cpx

                    pure (round opt,[i | (i,si) <- M.assocs sol, si == 1])

    where vs = M.keys gr
          mkCtr :: IloModel -> M.IntMap IloBoolVar -> Int -> IO [IloRange]
          mkCtr mdl vars i = forM [ni | ni <- neighbors gr i, ni > i] $ \ni -> do
                                            ct <- newIloObject env
                                            setBounds ct (0,1)
                                            setLinearCoef ct (fromJust $ vars ^. at i) 1
                                            setLinearCoef ct (fromJust $ vars ^. at ni) 1
                                            mdl `add` ct
                                            pure ct
                





