
module Main_IS where


import Graph
import Types
import IS.MIP
import Utils

import Control.Monad
import Data.Function

import IPSolver
import qualified Data.List as L
import System.Directory





test logfile maxw siz prob l = do
    b <- doesFileExist logfile
    when (not b) writeHeader 

    env <- newIloEnv 
    forM [1..10] $ \_ -> do
        gr <- mkRandomGraph maxw prob siz
        opt <- fromIntegral . fst <$> solveIS env gr
        lex <- lextest gr env
        let greed = fromIntegral $ length $ isGreed gr
            ret@(_,_,_,lexerr, greederr) = (opt, lex, greed, (opt-lex)/opt, (opt-greed)/opt)
        print ret
        appendFile logfile $ show maxw ++ ";" ++ show siz ++ ";" ++ show prob ++ ";" ++ show l ++ ";"
        appendFile logfile $ show opt ++ ";" ++ show lex ++ ";" ++ show greed ++ ";" ++ show lexerr ++ ";" ++ show greederr ++ "\n"
        pure ret
        
 
  where lextest gr env = do
            let coals = [ci | i <- [1..l], ci <- allSubsetsN i [1..siz]]
            vals <- fmap fst <$> forM coals (\ci -> solveIS env $ subGraph gr ci) :: IO [Int]
            let order = mkOrder $ zip vals (mkCoalition <$> coals)
                lexret = snd <$>lexCell [1..siz] order
                sol = glouton gr lexret
            pure $ fromIntegral $ length sol 
        greedy gr = let f v = 1 / fromIntegral (degree  gr v) 
                    in fromIntegral $ length $ glouton gr $ L.sortBy (compare `on` f) [1..siz]
            
        writeHeader = writeFile logfile "opt; lex; greed; lex_error; greed_error\n"
             
             
