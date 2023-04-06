module IS.MIP where


import Graph
import IPSolver
import qualified Data.Array as A
import qualified Data.IntMap as M
import Control.Monad



testMIP = do
    gr <- mkRandomGraph 20 0.5 10
    putStrLn $ unlines [show as | as <- M.assocs gr]
    env <- newIloEnv
    ret <- solveIS env gr
    putStrLn $ "cpx= " ++ show ret
    putStrLn $ "algo=" ++ show (isSolver gr)

{-| WARNING: vertices must be indiced from one to n -}

solveIS :: IloEnv -> Graph -> IO [Int]
solveIS env gr = do
        mdl <- newIloObject env
        cpx <- newIloObject env
        cpx `extract` mdl

        obj <- newIloObject env
        setMaximize obj >> mdl `add` obj

        vars <- A.listArray (1,length vs) <$> sequence [newIloObject env | vi <- vs]
        ctrs <- sequence [mkCtr mdl vars vi | vi <- vs]
        obj <- sequence [setLinearCoef obj (vars A.! vi) (fromIntegral $ _fromWeight $ score gr [vi]) | vi <- vs]
        
        cpx `exportModel` "test.lp"
        
        solve cpx

        sol <- sequence $ fmap (cpx `getValue`) vars

        pure [i | (i,si) <- A.assocs sol, si == 1]

        



    where vs = M.keys gr
          mkCtr :: IloModel -> A.Array Int IloBoolVar -> Int -> IO [IloRange]
          mkCtr mdl vars i = forM [ni | ni <- neighbors gr i, ni > i] $ \ni -> do
                                            ct <- newIloObject env
                                            setBounds ct (0,1)
                                            setLinearCoef ct (vars A.! i) 1
                                            setLinearCoef ct (vars A.! ni) 1
                                            mdl `add` ct
                                            pure ct
                



