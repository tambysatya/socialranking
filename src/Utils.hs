module Utils where



import qualified Data.Array as A 


getRow :: A.Array (Int,Int) a -> Int -> [a]
getRow arr i = [arr A.! (i,j) | j <- [1..jmax]]
    where (_,(_,jmax)) = A.bounds arr


mkVector :: [a] -> A.Array Int a
mkVector l = A.listArray (1, length l) l
mkArrayFromRows :: [[a]] -> A.Array (Int,Int) a
mkArrayFromRows l =  A.array ((1,1),(nconstraints,nvars)) arr
  where nconstraints = length l
        nvars = length $ head l
        arr = [((i,j), val) | (i, colvals) <- zip [1..] l, (j,val) <- zip [1..] colvals]


avg l = sum l / fromIntegral (length l)
