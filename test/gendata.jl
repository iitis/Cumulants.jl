using Distributions
using NPZ

srand(42)
cm = [[1. 0.7 0.7 0.7];[0.7 1. 0.7 0.7]; [0.7 0.7 1. 0.7]; [0.7 0.7 0.7 1]]
p = MvTDist(14, [0., 0., 0., 0.],cm)
x = transpose(rand(p, 75000000))
npzwrite("data/testdata.npz", x)
