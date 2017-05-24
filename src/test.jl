using SymmetricTensors
using Cumulants
a = randn(10000,42)
procs()

@time c = cumulants(a, 4)

addprocs(3)
procs()

@everywhere using Cumulants

@time c1 = cumulants(a, 4);

maximum(abs(convert(Array, c[3])- convert(Array, c1[3])))
