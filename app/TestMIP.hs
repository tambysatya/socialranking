module TestMIP where

import Utils
import Graph
import Types
import Plot 
import IS.MIP
import IS.Coalitions

import qualified Data.IntMap as M
import qualified Data.List as L
import qualified Data.Set as S
import System.Random.Shuffle
import Control.Monad
import System.Directory
import System.Random
import IPSolver


mainMIP = do
   let maxlens = [5,10,25,50]
       nits = [10,100,500,1000]
       siz = 100
   curves <- forM maxlens $ \maxlen -> do
                curve <- forM nits $ \nit -> do
                        err <- mainMIP' siz nit maxlen
                        pure (fromIntegral nit,err)
                pure (show maxlen, curve)
   plotError "number of coalitions" "ε-gap" "Figures/miperror_exactly.png" curves
        
    

{-| Average on 10 instances -}
mainMIP' :: Int ->  Int -> Int -> IO Double
mainMIP' = mainExactlyN

mainExactlyN :: Int -- size
               -> Int -- number of iterations
               -> Int -- max length
               -> IO Double
mainExactlyN siz nit maxlen = do
    env <- newIloEnv
    mkMainMIP (\gr -> testExactlyN env gr nit maxlen)  siz

mainAtMostN :: Int -- size
              -> Int -- number of iterations
              -> Int -- max length
              -> IO Double
mainAtMostN siz nit maxlen = do
    env <- newIloEnv
    mkMainMIP (\gr -> testAtMostN env gr nit maxlen)  siz

 
    

testAtMostN :: IloEnv
            -> Graph 
            -> Int  -- nit
            -> Int  -- maxlen
            -> IO (Int,Int,Int)
testAtMostN env gr nit maxlen = do
    coals <- genAtMostN gr nit maxlen
    putStrLn $ "generated " ++ show (S.size coals) ++ "/" ++ show nit
    mkTest env gr coals
    

testExactlyN :: IloEnv
            -> Graph 
            -> Int  -- nit
            -> Int  -- maxlen
            -> IO (Int,Int,Int)
testExactlyN env gr nit maxlen = do
    coals <- genExactlyN gr nit maxlen
    putStrLn $ "generated " ++ show (S.size coals) ++ "/" ++ show nit
    mkTest env gr coals


{- Builders -}

{-| Computes the average error on 10 instances of size siz
    The function must return (opt, glout, rnd) TODO typer
-}
mkMainMIP :: (Graph -> IO (Int,Int,Int)) -> Int -> IO Double
mkMainMIP testfun siz = do
    let  
         prob = 0.5
         maxw = 20
         
    env <- newIloEnv
    ret <- forM [1..10] $ \_ -> do
            gr <- mkRandomGraph maxw prob siz
            testfun gr
    let (opts,glouts,rnds) = unzip3 ret
        avgI l = fromIntegral (sum l) / fromIntegral (length l)
        avgD l = sum l / fromIntegral (length l)
        [_opts,_glouts,_rnds]= avgI <$> [opts,glouts,rnds]
        err = avgD $ fmap (\x -> 1-x) $ zipWith (/) (fromIntegral <$> glouts) ((fromIntegral <$> opts) :: [Double])
        result = (_opts,_glouts,_rnds, err)
    print result
    pure err
 
{-| Computes  (opt,glout,rnd) given a graph and a set of coalitions -}
    
mkTest :: IloEnv -> Graph -> S.Set (Coalition Int) -> IO (Int,Int,Int)
mkTest env gr coals = do
    let  scoremip :: Coalition Int -> IO Int
         scoremip coal = fst <$> solveIS env (subGraph gr $ toList coal)
    order <- mkOrderIO scoremip $ S.toList coals
    putStrLn $ "computed"
    rnd <- do
        ord <- shuffleM [1..n]
        pure $ glouton gr ord
    (optval,opt) <- solveIS env gr

    let rank = reverse $ snd <$> lexCell [1..n] order
        glout = glouton gr rank

        rndval = _fromWeight $ score gr rnd
        gloutval = _fromWeight $ score gr glout
    print (optval,gloutval,rndval)
    pure (optval,gloutval,rndval)
 where n = M.size gr


