module Main (main) where

import Graph
import Types
import qualified Data.IntMap as M
import qualified Data.List as L
import System.Random.Shuffle
import Control.Monad
import System.Directory

logfile = "log.txt"

main :: IO ()
main = do
  b <- doesFileExist logfile
  when (not b) $ writeHeader logfile
  forM_ configs $ \(wi, prob, perc,i) -> do 
    r <- test wi prob 10 perc
    appendFile logfile $ show r ++ "\n"
    print r
    
   where wmax = [10,20,30]
         probs = [0.25,0.5,0.75]
         percentages = [0.01,0.05,0.001,0.005]++[0.1,0.2..1]
         ninstances = 10
         configs = [(wi,prob,perc,i) | wi <- wmax, prob <- probs, perc <- percentages, i <- [1..ninstances]]


data Result = Result  {
                        _graphSiz :: Int,
                        _wmax :: Int,
                        _graphProb :: Double,
                        _percentageTaken :: Double,
                        _optVal :: Int,
                        _lexVal :: Int,
                        _randVal :: Int
                      }

truncateNdigits k n = (fromIntegral $ round $ n*10^k) / 10^k


mkISOrder :: Graph -> [[Int]] -> TotalPreorder (Coalition Int)
mkISOrder gr coals = mkCoalitionOrder (_fromWeight . score gr) (isSolver . subGraph gr <$> coals)

order gr = mkISOrder gr $ allSubsets [1..n] 
    where n = M.size gr
randomOrder gr len = do
        selected <- take len <$> shuffleM candidates
        pure $ mkISOrder gr selected
    where n = M.size gr
          candidates = allSubsets [1..n]
glouton :: Graph -> [Int] -> [Int]
glouton gr [] = []
glouton gr (c:candidates) = c:glouton gr (candidates L.\\ neighbors gr c)

randomSol :: Graph -> IO [Int]
randomSol gr = glouton gr <$> shuffleM (M.keys gr)

test :: Int -> Double -> Int -> Double -> IO Result
test wmax prob siz percentage = do
    gr <- mkRandomGraph (Weight wmax) prob siz
    rnd <- randomSol gr
    
    datas <- if percentage == 1 then pure $ order gr
                                else randomOrder gr totake

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





writeHeader :: String -> IO ()
writeHeader str = writeFile str "size;wmax;prob;perc;opt;lex;rnd\n"
instance Show Result where
    show (Result gsiz' wmax' gprob totake opt' lex' rnd') = L.intercalate ";" $ fmap show [gsiz,wmax, truncateNdigits 2 gprob, truncateNdigits 3 totake, opt, lex, rnd]
                where [gsiz,wmax, opt,lex,rnd] = fmap fromIntegral [gsiz',wmax', opt', lex', rnd']
