module Plot where

import Graphics.Gnuplot.Simple

import Graph
import Types
import qualified Data.IntMap as M
import qualified Data.List as L
import qualified Data.Set as S
import System.Random.Shuffle
import Control.Monad
import System.Directory
import System.Random
import Graphics.Gnuplot.Simple


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



plotError :: String -> String -> String -> [(String,[(Double,Double)])] -> IO ()
plotError xlabel ylabel fname vals = plotPathsStyle attr [(defaultStyle {lineSpec = CustomStyle [LineTitle name]},pts) | (name,pts) <- vals]
    where attr = [PNG fname, XLabel xlabel, YLabel ylabel]

rError :: Result -> Double
rError r = fromIntegral (_optVal r - _lexVal r) / fromIntegral (_optVal r)

avgError:: [Result] -> Double
avgError rs = sum (fmap rError rs) / fromIntegral (length rs)

plotErrorIO :: String -> [(String,[(Double,[Result])])] -> IO ()
plotErrorIO fname results = plotError "coalitions taken" "%error" fname  $ fmap (\(n,pts) -> (n,[(xi, avgError yis) | (xi,yis) <- pts])) results-- [(prop, avgError res) | (name,pts) <- results, (prop,res) <- pts]



writeHeader :: String -> IO ()
writeHeader str = writeFile str "size;wmax;prob;perc;opt;lex;rnd\n"
instance Show Result where
    show (Result gsiz' wmax' gprob totake opt' lex' rnd') = L.intercalate ";" $ fmap show [gsiz,wmax, truncateNdigits 2 gprob, totake, opt, lex, rnd]
                where [gsiz,wmax, opt,lex,rnd] = fmap fromIntegral [gsiz',wmax', opt', lex', rnd']
