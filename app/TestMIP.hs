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
   let siz = 300
       maxlens = [50,100,150] --[5,10,15] -- [5,10,25]

       nits = [10,1000, 2000, 3000] --,2000,5000] -- [10,100,250,500,750,1000,1250,1500]
       
       maxw = 20
       density = 0.5
       gparam = GraphParam siz maxw density
       instype = "exacts"
   curves <- forM maxlens $ \maxlen -> do
                datas <- forM nits $ \nit -> do
                        (epsgap,errsel) <- mainMIP' gparam nit maxlen
                        pure ((fromIntegral nit,epsgap), (fromIntegral nit, errsel))
                let (curveseps, curvessel) = unzip datas
                writeFile (paramToFile ("datas/err-" ++ path instype maxlen) gparam) $  show curveseps -- epsgap       
                writeFile (paramToFile ("datas/sel-" ++ path instype maxlen) gparam) $  show curvessel -- selection error
                pure ((show maxlen, curveseps), (show maxlen, curvessel))
   let (eps,sel) = unzip curves
   plotError "number of coalitions" "% error" ("Figures/mipsel_"++instype++"-" ++ paramToFile "" gparam ++ ".png") sel
   plotError "number of coalitions" "ε-gap" ("Figures/mipeps_"++instype++"-" ++ paramToFile "" gparam ++ ".png") eps
         
    

{-| Average on 10 instances -}
mainMIP' = mainExactlyN
-- mainMIP' = mainAtMostN 

mainExactlyN :: GraphParam
               -> Int -- number of iterations
               -> Int -- max length
               -> IO (Double,Double) -- (percentage error, percentage wrong sel)
mainExactlyN params nit maxlen = do
    env <- newIloEnv
    mkMainMIP (\gr -> testExactlyN env gr nit maxlen) params

mainAtMostN :: GraphParam
              -> Int -- number of iterations
              -> Int -- max length
              -> IO (Double,Double) -- (percentage error, percentage wrong sel)
mainAtMostN params nit maxlen = do
    env <- newIloEnv
    mkMainMIP (\gr -> testAtMostN env gr nit maxlen) params

 
    

testAtMostN :: IloEnv
            -> Graph 
            -> Int  -- nit
            -> Int  -- maxlen
            -> IO (Int,Int,Int,Double)
testAtMostN env gr nit maxlen = do
    coals <- genAtMostN gr nit maxlen
    putStrLn $ "generated " ++ show (S.size coals) ++ "/" ++ show nit
    mkTest env gr coals
    

testExactlyN :: IloEnv
            -> Graph 
            -> Int  -- nit
            -> Int  -- maxlen
            -> IO (Int,Int,Int,Double)
testExactlyN env gr nit maxlen = do
    coals <- genExactlyN gr nit maxlen
    putStrLn $ "generated " ++ show (S.size coals) ++ "/" ++ show nit
    mkTest env gr coals


{- Builders -}

{-| Computes the average error on 10 instances of size siz
    The function must return (opt, glout, rnd) TODO typer
-}
mkMainMIP :: (Graph -> IO (Int,Int,Int,Double)) -> GraphParam -> IO (Double,Double)
mkMainMIP testfun (GraphParam siz maxw prob) = do
    env <- newIloEnv
    ret <- forM [1..10] $ \_ -> do
            gr <- mkRandomGraph maxw prob siz
            testfun gr
    let (opts,glouts,rnds,psuccess) = unzip4 ret
        avgI l = fromIntegral (sum l) / fromIntegral (length l)
        avgD l = sum l / fromIntegral (length l)
        [_opts,_glouts,_rnds]= avgI <$> [opts,glouts,rnds]
        err = avgD $ fmap (\x -> 1-x) $ zipWith (/) (fromIntegral <$> glouts) ((fromIntegral <$> opts) :: [Double])
        errsel = avgD $ fmap (\x -> 1-x) psuccess
        result = (_opts,_glouts,_rnds, err, errsel)
    print result
    pure (err,errsel)

unzip4 :: [(a,b,c,d)] -> ([a],[b],[c],[d])
unzip4 [] = error "unzip4: empty 4-upple"
unzip4 [(a,b,c,d)] = ([a],[b],[c],[d])
unzip4 ((a,b,c,d):xs) = let (as,bs,cs,ds) = unzip4 xs
                        in (a:as,b:bs,c:cs,d:ds)
unzip4 _ = error "unzip4: must contain a 4-upple"
 
{-| Computes  (opt,glout,rnd) given a graph and a set of coalitions -}
    
mkTest :: IloEnv -> Graph -> S.Set (Coalition Int) -> IO (Int,Int,Int,Double)
mkTest env gr coals = do
    let  scoremip :: Coalition Int -> IO Int
         scoremip coal = fst <$> solveWeightedIS env (subGraph gr $ toList coal)
    order <- mkOrderIO scoremip $ S.toList coals
    putStrLn $ "computed"
    rnd <- do
        ord <- shuffleM [1..n]
        putStrLn "running glouton"
        pure $ glouton gr ord
    putStr "solving the mip..."
    (optval,opt) <- solveWeightedIS env gr
    putStrLn "done"

    let rank = reverse $ snd <$> lexCell [1..n] order
        glout = glouton gr rank

        rndval = _fromWeight $ score gr rnd
        gloutval = _fromWeight $ score gr glout
        reussite = propReussite opt glout
    print (optval,gloutval,rndval)
    pure (optval,gloutval,rndval, reussite)
 where n = M.size gr

-- calculerla proportion de réussite: la proportion de sommets correctement selectionnés
-- inconvenient: privilegie de prendre le maximum de sommets
propReussite optsol glout = fromIntegral correct / siz
    where correct = length [gi | gi <- glout, gi `elem` optsol] 
          siz = fromIntegral $ length glout


