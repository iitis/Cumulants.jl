using FactCheck
using SymmetricTensors
using Cumulants
using Distributions
using NullableArrays
using Iterators
import Cumulants: indpart, momentel, momentseg
import SymmetricTensors: indices

include("test_helpers/s_naive.jl")
include("test_helpers/naivecum.jl")


gaus_dat =  [[-0.88626   0.279571];
            [-0.704774  0.131896]]

data = clcopulagen(10, 4)

facts("Helper functions") do
  context("center") do
    @fact sum(abs(mean(center(data), 1))) --> roughly(0, 1e-15)
  end
  context("momentel")do
    @fact momentel([1.,2.,3.], [4.,5.,6.], [7.,8.,9.]) --> 90.
    m = collect(reshape(1.:4., 2, 2))
    @fact momentseg((2,2), m ,m) --> [2.5 5.5; 5.5 12.5]
  end
  context("indpart") do
    @fact indpart(4,2)[1] --> [[[1,2],[3,4]], [[1,3],[2,4]], [[1,4],[2,3]]]
    @fact indpart(4,2)[2] --> [[2,2],[2,2],[2,2]]
    @fact indpart(4,2)[3] --> 3
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

facts("Exceptions") do
  context("Size of blocks") do
    @fact_throws DimensionMismatch, momentbs(data, 4,  25)
    @fact_throws DimensionMismatch, cumulants(data, 3,  25)
  end
end

facts("Cumulants vs naive implementation") do
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
  cn2, cn3, cn4, cn5, cn6, cn7, cn8 = snaivecumulant(gaus_dat, 8)
  @fact cn2 --> roughly(naivecumulant(gaus_dat, 2))
  @fact cn3 --> roughly(zeros(Float64, 2,2,2))
  @fact cn4 --> roughly(zeros(Float64, 2,2,2,2), 1e-3)
  @fact cn5--> roughly(zeros(Float64, 2,2,2,2,2))
  @fact cn6 --> roughly(zeros(Float64, 2,2,2,2,2,2), 1e-4)
  @fact cn7 --> roughly(zeros(Float64, 2,2,2,2,2,2,2))
  @fact cn8 --> roughly(zeros(Float64, 2,2,2,2,2,2,2,2), 1e-5)
end

cn2, cn3, cn4, cn5, cn6, cn7, cn8 = snaivecumulant(data[:, 1:2], 8)
facts("Cumulants vs semi-naive square") do
  c2, c3, c4, c5, c6, c7, c8 = cumulants(data[:, 1:2], 8, 2)
  @fact convert(Array, c2) --> roughly(cn2)
  @fact convert(Array, c3) --> roughly(cn3)
  @fact convert(Array, c4) --> roughly(cn4)
  @fact convert(Array, c5) --> roughly(cn5)
  @fact convert(Array, c6) --> roughly(cn6)
  @fact convert(Array, c7) --> roughly(cn7)
  @fact convert(Array, c8) --> roughly(cn8)
end
