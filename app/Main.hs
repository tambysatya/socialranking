module Main (main) where

import Graph
import Types
import Plot 

import qualified Data.IntMap as M
import qualified Data.List as L
import qualified Data.Set as S
import System.Random.Shuffle
import Control.Monad
import System.Directory
import System.Random


main :: IO ()
main = plotEasy --mainLarge


mainLarge = do 
  b <- doesFileExist logfile
  when (not b) $ writeHeader logfile
  forM_ configs $ \(wi,density,(siz,totake,accept),i) -> do 
    r <- test wi density siz totake accept
    appendFile logfile $ show r ++ "\n"
    print r

  {-
  forM_ configs $ \(wi, prob, (size,perc),i) -> do 
    r <- test wi prob size perc
    appendFile logfile $ show r ++ "\n"
    print r
    -}
   where 
         wmax = [10,20,30]
         probs = [0.25,0.5,0.75]
         -- percentages = [(20,0.0001), (30,1e-7), (40,1e-10)]
         instances = [(20,0.0001,0.8), -- (size, percentages taken, acceptpercentage) -- The accept percentage influence the difficulty of the problems to be solved
                      (30,1e-7,0.5),
                      (40,1e-10,0.1)]

         ninstances = 10
         configs = [(wi,density,ins,i) | wi <- wmax, density <- probs, ins <- instances, i <- [1..ninstances]]
         -- configs = [(wi,prob,perc,i) | wi <- wmax, prob <- probs, perc <- percentages, i <- [1..ninstances]]
         logfile = "loghard.txt"


mainEasy = do
  b <- doesFileExist logfile
  when (not b) $ writeHeader logfile
  forM_ configs $ \(wi, prob, perc,i) -> do 
    r <- test wi prob 10 perc 0.5
    appendFile logfile $ show r ++ "\n"
    print r
    
   where wmax = [20]
         probs = [0.25,0.5,0.75]
         percentages = [0.01,0.1,1] -- 0.001,0.005]++[0.1,0.2..1]
         -- percentages = [0.01,0.05,0.001,0.005]++[0.1,0.2..1]
         ninstances = 10
         configs = [(wi,prob,perc,i) | wi <- wmax, prob <- probs, perc <- percentages, i <- [1..ninstances]]
         logfile = "logeasy.txt"

plotEasy = do
  b <- doesFileExist logfile
  when (not b) $ writeHeader logfile
  results <- forM densities $ \di -> do 
      res <- forM percentages $ \pi -> do --forM_ configs $ \(wi, prob, perc,i) -> do 
        r <- forM [1..10] $ \i -> do
                r <- test 20 di size pi 0.5
                appendFile logfile $ show r ++ "\n"
                print r
                pure r
        pure (totake pi,r)
      plotErrorIO  ("Figures/error_" ++ show di ++ ".png") [(show di ++ " % edges",res)]
      pure (show di ++ " % edges", res)
  pure ()
  plotErrorIO ("Figures/error.png") results
    
   where wmax = [10,20,30]
         densities = [0.25,0.5,0.75]
         --percentages = [0.01,0.05,0.001,0.005]++[0.1,0.2..1]
         percentages = [1e-5,2e-5..1e-3]
         ninstances = 10
         -- configs = [(wi,prob,perc,i) | wi <- wmax, prob <- probs, perc <- percentages, i <- [1..ninstances]]
         configs = []
         logfile = "ploteasy.txt"
         size = 17
         totake pi = fromIntegral (truncate $ pi * (2^size-1))



mkISOrder :: Graph -> [[Int]] -> TotalPreorder (Coalition Int)
mkISOrder gr coals = mkCoalitionOrder (_fromWeight . score gr) (isSolver . subGraph gr <$> coals)
test :: Int  -- max weight to be sampled
     -> Double  -- density (% of two vertices to be connected)
     -> Int     -- number of vertices
     -> Double  -- proportion of coalitions that are evaluated
     -> Double -- Proportion of acceptation of each vertex
     -> IO Result
test wmax prob siz percentage acceptp = do
    gr <- mkRandomGraph (Weight wmax) prob siz
    rnd <- randomSol gr
    
    datas <- if percentage == 1 then pure $ order gr
                                else do 
                                    putStrLn $ "taking " ++ show totake
                                    -- subset <- fasterRandomOrder gr totake acceptp S.empty -- TODO
                                    randomOrder gr totake
                                    --putStrLn "found order"
                                    --pure $ mkISOrder gr $ S.toList subset 

    let opt = isSolver gr
        ordret = reverse $ lexCell [1..siz] datas
        ord = fmap snd ordret
        glout = glouton gr ord
        scoreopt = _fromWeight $ score gr opt
        scoreglout = _fromWeight $ score gr glout
        scorernd = _fromWeight $ score gr rnd

        

    print (scoreopt,scoreglout,scorernd)
      --putStrLn $ "opt=" ++ show opt ++ " lexcell=" ++ show glout ++  " order=" ++ show ord
 --   putStrLn $ "total order=" ++ unlines (fmap show  ordret)
--    putStrLn $ "random=" ++ show rnd ++ " val= " ++ show scorernd

    pure $ Result siz wmax prob percentage scoreopt scoreglout scorernd


  where ncoal = 2^siz - 2
        totake = fromIntegral (truncate $ percentage * ncoal)




order gr = mkISOrder gr $ allSubsets [1..n] 
    where n = M.size gr
randomOrder gr len = do
        selected <- take len <$> shuffleM candidates
        pure $ mkISOrder gr selected
    where n = M.size gr
          candidates = allSubsets [1..n]
fasterRandomOrder :: Graph -> Int -> Double -> S.Set [Int] -> IO (S.Set [Int])
fasterRandomOrder _ 0 _ s = pure s
fasterRandomOrder gr len keepprob set = do
    keep <- forM [1..len] $ \_ -> randomRIO (0,1)
    let entry = [i | (i,keepi) <- zip [1..] keep, keepi <= keepprob]
    if entry `S.member` set then fasterRandomOrder gr len keepprob set 
                            else fasterRandomOrder gr (len-1) keepprob (S.insert entry set) 

        
glouton :: Graph -> [Int] -> [Int]
glouton gr [] = []
glouton gr (c:candidates) = c:glouton gr (candidates L.\\ neighbors gr c)

randomSol :: Graph -> IO [Int]
randomSol gr = glouton gr <$> shuffleM (M.keys gr)




