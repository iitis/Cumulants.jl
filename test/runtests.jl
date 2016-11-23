using FactCheck
using SymmetricTensors
using Cumulants
using Distributions
using NullableArrays
using Iterators

include("test_helpers/s_naive.jl")
include("test_helpers/naivecum.jl")


gaus_dat =  [[-0.88626   0.279571];
            [-0.704774  0.131896]]

data = clcopulagen(10, 4)

facts("Helper functions") do
  context("center") do
    @fact sum(abs(mean(center(data), 1))) --> roughly(0, 1e-15)
  end
end
facts("Moments") do
  context("3") do
    @fact convert(Array, momentbs(data, 3, 2)) --> roughly(moment_n(data, 3))
  end

  context("4") do
    @fact convert(Array, momentbs(data, 4, 2)) --> roughly(moment_n(data, 4))
  end
end

facts("Comulants vs naive implementation") do
  cn = [naivecumulant(data, i) for i = 2:6]
  context("Square blocks") do
    c2, c3, c4, c5, c6 = cumulants(data, 6, 2)
    @fact convert(Array, c2) --> roughly(cn[1])
    @fact convert(Array, c3) --> roughly(cn[2])
    @fact convert(Array, c4) --> roughly(cn[3])
    @fact convert(Array, c5) --> roughly(cn[4])
    @fact convert(Array, c6) --> roughly(cn[5])
  end

  context("Non-square blocks") do
    c2, c3, c4, c5, c6 = cumulants(data[:, 1:3], 6, 2)
    @fact convert(Array, c2) --> roughly(cn[1][fill(1:3, 2)...])
    @fact convert(Array, c3) --> roughly(cn[2][fill(1:3, 3)...])
    @fact convert(Array, c4) --> roughly(cn[3][fill(1:3, 4)...])
    @fact convert(Array, c5) --> roughly(cn[4][fill(1:3, 5)...])
    @fact convert(Array, c6) --> roughly(cn[5][fill(1:3, 6)...])
  end
end

facts("test semi-naive against gaussian") do
  cg = snaivecumulant(gaus_dat, 8)
  @fact cg["c2"] --> roughly(naivecumulant(gaus_dat, 2))
  @fact cg["c3"] --> roughly(zeros(Float64, 2,2,2))
  @fact cg["c4"] --> roughly(zeros(Float64, 2,2,2,2), 1e-3)
  @fact cg["c5"] --> roughly(zeros(Float64, 2,2,2,2,2))
  @fact cg["c6"] --> roughly(zeros(Float64, 2,2,2,2,2,2), 1e-4)
  @fact cg["c7"] --> roughly(zeros(Float64, 2,2,2,2,2,2,2))
  @fact cg["c8"] --> roughly(zeros(Float64, 2,2,2,2,2,2,2,2), 1e-5)
end

facts("Cumulants vs semi-naive non-square") do
  c2, c3, c4, c5, c6, c7, c8 = cumulants(data[:, 1:2], 8, 2)
  cnn = snaivecumulant(data[:, 1:2], 8)
  @fact convert(Array, c2) --> roughly(cnn["c2"])
  @fact convert(Array, c3) --> roughly(cnn["c3"])
  @fact convert(Array, c4) --> roughly(cnn["c4"])
  @fact convert(Array, c5) --> roughly(cnn["c5"])
  @fact convert(Array, c6) --> roughly(cnn["c6"])
  @fact convert(Array, c7) --> roughly(cnn["c7"])
  @fact convert(Array, c8) --> roughly(cnn["c8"])
end
