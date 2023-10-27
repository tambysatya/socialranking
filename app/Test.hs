module Test where

import IPSolver  (IloEnv, newIloEnv)
import Control.Monad
import Lexcel
import System.Directory
import qualified Data.List as L
import Utils
import MIP.Class
import MIP.KP
import MIP.IS



testInstance :: Problem p => (String -> IO ()) -> IloEnv -> Int -> Int -> Maybe Int -> p -> IO (Double,Double,Double,Double, Double, Double)
testInstance writeLog env nclasses ncoals coalsizeM pb = do
    (lex, maxval) <- lexcelGreedy nclasses ncoals coalsizeM pb
    (blex, bmaxval) <- biasedLexcelGreedy nclasses ncoals coalsizeM pb
    opt <- fromIntegral.round <$> solve env pb
    let lexratio = precision 2 $ lex/opt*100
        maxratio = precision 2 $ maxval/opt*100
        hratio = precision 2 $ heuristic/opt*100

        blexratio = precision 2 $ blex/opt*100
        bmaxratio = precision 2 $ bmaxval/opt*100
    
    putStrLn $ "opt="++show opt ++ " lex=" ++ show lex  ++ " biased lex=" ++ show blex ++ " greedy=" ++ show heuristic ++ " max=" ++ show maxval ++ " biased max=" ++ show bmaxval
    writeLog $ show lexratio ++ ";" ++ show blexratio ++ ";" ++ show hratio ++ ";" ++ show maxratio ++ ";" ++ show bmaxratio
    pure (opt,lexratio,blexratio,hratio,maxratio,bmaxratio)
  where heuristic = greedy pb


testKP :: Int -> Int -> Int -> Int -> Int -> Maybe Int -> Int -> IO ()
testKP nitems nctrs maxval nclasses ncoals coalsizeM ntests = do
    env <- newIloEnv
    existsP <- doesFileExist logKP
    when (not existsP) $ appendFile logKP header
    appendFile logKP comment
    results <- forM [1..ntests] $ \_ -> do
        pb <- generateUniformFeasibleKP nitems nctrs maxval
        testInstance writeLog env nclasses ncoals coalsizeM pb
    appendFile logKP "\n"
    let (opts,lexs,blexs,hs,maxs,bmaxs) = L.unzip6 results
        [lexavg,blexavg,havg,maxavg,bmaxavg] = fmap avg [lexs,blexs,hs,maxs,bmaxs]
    putStrLn "---"
    putStrLn $ "opt=" ++ show (avg opts)
    putStrLn $ "lex=" ++ show (avg lexs)
    putStrLn $ "biased lex=" ++ show (avg blexs)
    putStrLn $ "greedy=" ++ show (avg hs)
    putStrLn $ "maxs=" ++ show (avg maxs)
    putStrLn $ "biased maxs=" ++ show (avg bmaxs)

    appendFile logKP $ "lex=" ++ show (precision 3 lexavg) ++ " biased lex=" ++ show (precision 3 blexavg) ++ " heuristic=" ++ show (precision 3 havg) ++ " max_score=" ++ show (precision 3 maxavg) ++ " biased_max=" ++ show (precision 3 bmaxavg) ++ "\n\n"

    pure ()
  where prefix = show nitems ++ ";" ++ show nctrs ++ ";" ++ show maxval ++ ";" ++ show nclasses ++ ";" ++ show ncoals ++ ";" ++ show coalsize ++ ";"
        header = "nitems;nctrs;wmax;nclasses;ncoals;coalsize;lex;biased;heuristic;lexmax;biasedmax\n\n"
        comment = "# nitems=" ++ show nitems ++ " nctrs=" ++ show nctrs ++ " maxval=" ++ show maxval ++ " nclasses=" ++ show nclasses ++ " ncoals=" ++ show ncoals ++ " coalsizeM=" ++ show coalsize ++ "\n"
        writeLog str = appendFile logKP $ prefix ++ str ++ "\n"
        coalsize = case coalsizeM of
                                Nothing -> nitems
                                Just x -> x
        logKP = "logs/kp"++show nitems ++ "_" ++ show nctrs




testIS :: Int -> Int -> Double -> Int -> Int -> Maybe Int -> Int -> IO ()
testIS nvertices maxval density nclasses ncoals coalsizeM ntests = do
    env <- newIloEnv
    existsP <- doesFileExist logIS
    when (not existsP) $ writeFile logIS header
    appendFile logIS comment
    results <- forM [1..ntests] $ \_ -> do
        pb <- generateIS nvertices maxval density
        testInstance writeLog env nclasses ncoals coalsizeM pb
    appendFile logIS "\n"
    let (opts,lexs,blexs,hs,maxs,bmaxs) = L.unzip6 results
        [lexavg,blexavg,havg,maxavg,bmaxavg] = fmap avg [lexs,blexs,hs,maxs,bmaxs]
    putStrLn "---"
    putStrLn $ "opt=" ++ show (avg opts)
    putStrLn $ "lex=" ++ show (avg lexs)
    putStrLn $ "biased lex=" ++ show (avg blexs)
    putStrLn $ "greedy=" ++ show (avg hs)
    putStrLn $ "maxs=" ++ show (avg maxs)
    putStrLn $ "biased maxs=" ++ show (avg bmaxs)
    appendFile logIS $ "lex=" ++ show (precision 3 lexavg) ++ " biased lex=" ++ show (precision 3 blexavg) ++ " heuristic=" ++ show (precision 3 havg) ++ " max_score=" ++ show (precision 3 maxavg) ++ " biased_max=" ++ show (precision 3 bmaxavg) ++ "\n\n"
    pure ()
  where prefix = show nvertices ++ ";" ++ show maxval ++ ";" ++ show density ++ ";" ++ show nclasses ++ ";" ++ show ncoals ++ ";" ++ show coalsize ++ ";"
        comment = "# nvertices=" ++ show nvertices ++ " maxval=" ++ show maxval ++ " density=" ++ show density ++ " nclasses=" ++ show nclasses ++ " ncoals=" ++ show ncoals ++ " coalsize=" ++ show coalsize ++ "\n"
        header = "nvertices;wmax;density;nclasses;ncoals;coalsize;lex;biased;heuristic;lexmax;biasedmax\n\n"
        writeLog str = appendFile logIS $ prefix ++ str ++ "\n"
        coalsize = case coalsizeM of
                                Nothing -> nvertices
                                Just x -> x
        logIS = "logs/is" ++ show nvertices ++ "_" ++ show maxval





precision d n = mul / 10^d
    where mul = fromIntegral $ truncate $ n*10^d
