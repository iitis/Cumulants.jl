using FactCheck
using SymmetricTensors
using Cumulants
using Distributions
using NullableArrays
using Iterators
import Cumulants: indpart, momentseg, splitdata, mom_el, accesscum, outprodblocks,
 IndexPart, outerpodcum

import SymmetricTensors: indices

include("test_helpers/s_naive.jl")
include("test_helpers/naivecum.jl")


data = clcopulagen(10, 4)

facts("Helper functions") do
  context("splitdata") do
    M = [1. 2. 3. 4.; 5. 6. 7. 8.]
    @fact splitdata(M, 2) --> [[1. 2. ; 5. 6.],[3. 4. ; 7. 8.]]
    @fact splitdata(M, 3)[1] --> [1. 2. 3.; 5. 6. 7.]
  end
  context("moment helpers")do
    @fact mom_el([[1. 2. ; 5. 6.],[3. 4. ; 7. 8.]], (1,1)) --> 19.
    @fact mom_el([[1. 2. ; 5. 6.],[3. 4. ; 7. 8.]], (1,2)) --> 22.
    @fact momentseg([[1. 2.; 5. 6.],[3. 4.; 7. 8.]])-->[[19.0  22.0];[24.0  28.0]]
  end
end
facts("Moments") do
  context("2") do
    mom = moments([[1. 2. ; 5. 6.],[3. 4. ; 7. 8.]], 2).frame
    @fact mom[1,2].value --> [19.0 22.0; 24.0 28.0]
  end
  d = splitdata(data, 2)
  context("3") do
    @fact convert(Array, moments(d, 3)) --> roughly(moment_n(data, 3))
  end
  context("4") do
    @fact convert(Array, moments(d, 4)) --> roughly(moment_n(data, 4))
  end
end

facts("Exceptions") do
  context("Size of blocks") do
    @fact_throws DimensionMismatch, momentbs(data, 4,  25)
    @fact_throws DimensionMismatch, cumulants(data, 3,  25)
  end
end

facts("Cumulant helper functions") do
  indexpart = indpart(4,2)
  context("indpart") do
    @fact indexpart[1].part --> [[1,2],[3,4]]
    @fact indexpart[2].part --> [[1,3],[2,4]]
    @fact indexpart[3].part --> [[1,4],[2,3]]
  end
  context("operation on blocks") do
    c = SymmetricTensor([1.0 2.0 3.0; 2.0 4.0 6.0; 3.0 6.0 5.0])
    blocks = accesscum((1,1,1,1), indexpart[1], c)
    @fact blocks --> [[1.0 2.0; 2.0 4.0], [1.0 2.0; 2.0 4.0]]
    @fact accesscum((1,1,1,2), indexpart[1], c) --> [[1.0 2.0; 2.0 4.0],
    [3.0 0.0; 6.0 0.0]]
    @fact accesscum((1,1,1,2), indexpart[3], c) --> [[3.0 0.0; 6.0 0.0],
    [1.0 2.0; 2.0 4.0]]
    block = outprodblocks(indexpart[1], blocks)
    @fact block[:,:,1,1] --> [1.0  2.0; 2.0  4.0]
    @fact block[:,:,1,2] --> [2.0  4.0; 4.0  8.0]
    @fact outerpodcum(4,2,c).frame[1,1,1,1].value[1,1,:,] --> [3.0, 6.0, 6.0, 12.0]
  end
end

gaus_dat =  [[-0.88626   0.279571];
            [-0.704774  0.131896]]


facts("Cumulants vs naive implementation") do
  context("Test naive implentation") do
    @fact naivecumulant(gaus_dat, 3) --> roughly(zeros(Float64, 2,2,2))
  end
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
    c2, c3, c4, c5, c6 = cumulants(data, 6, 3)
    @fact convert(Array, c2) --> roughly(cn[1])
    @fact convert(Array, c3) --> roughly(cn[2])
    @fact convert(Array, c4) --> roughly(cn[3])
    @fact convert(Array, c5) --> roughly(cn[4])
    @fact convert(Array, c6) --> roughly(cn[5])
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
  @fact convert(Array, cumulants(gaus_dat, 3)[2]) --> roughly(zeros(Float64, 2,2,2))
  @fact convert(Array, c2) --> roughly(cn2)
  @fact convert(Array, c3) --> roughly(cn3)
  @fact convert(Array, c4) --> roughly(cn4)
  @fact convert(Array, c5) --> roughly(cn5)
  @fact convert(Array, c6) --> roughly(cn6)
  @fact convert(Array, c7) --> roughly(cn7)
  @fact convert(Array, c8) --> roughly(cn8)
end
