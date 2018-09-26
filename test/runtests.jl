using Test
using SymmetricTensors
using Cumulants
using Distributions
using Combinatorics
using Random
using Distributed
import Base: rand

import Cumulants: indpart, momentblock, blockel, accesscum, outprodblocks,
 IndexPart, outerprodcum, usebl, momel, mixel

import SymmetricTensors: pyramidindices

include("testfunctions/pyramidcumulants.jl")
include("testfunctions/mom2cum.jl")
include("testfunctions/leeuw_cumulants_no_nested_func.jl")

Random.seed!(42)
x = randn(10,4);
d = MvLogNormal(x'*x)
data = Array(rand(d, 10)')

@testset "Helper functions" begin
  @testset "moment helpers" begin
    M = [1. 2.  5. 6. ; 3. 4.  7. 8.]
    @test blockel(M, (1, 1), (1, 1), 2) == 5.0
    @test blockel(M, (1, 1), (2, 2), 2) == 37.0
    @test usebl((1, 1, 3), 5, 2, 3) == (2, 2, 1)
    @test momentblock(M, (1, 1), (2, 2), 2) == [[5.0 7.0]; [7.0 10.0]]
  end
end

@testset "Moment" begin
  M = [-0.88626  0.279571; -0.704774  0.131896]
  @testset "naivemoment" begin
    @test isapprox((naivemoment(M, 3))[:, :, 1], [-0.523092 0.142552; 0.142552 -0.0407653], atol=1.0e-5)
    @test isapprox((naivemoment(M, 3))[:, :, 2], [0.142552 -0.0407653; -0.0407653 0.0120729], atol=1.0e-5)
  end
  @testset "pyramidmoment" begin
    @test isapprox((pyramidmoment(M, 3))[:, :, 1], [-0.523092 0.142552; 0.142552 -0.0407653], atol=1.0e-5)
    @test isapprox((pyramidmoment(M, 3))[:, :, 2], [0.142552 -0.0407653; -0.0407653 0.0120729], atol=1.0e-5)
  end
  @testset "2" begin
    @test Array(moment(data, 2)) ≈ naivemoment(data, 2)
  end
  @testset "3" begin
    @test Array(moment(data, 3)) ≈ naivemoment(data, 3)
  end
  @testset "4" begin
    @test Array(moment(data, 4)) ≈ naivemoment(data, 4)
    @test Array(moment(data, 4, 3)) ≈ naivemoment(data, 4)
  end
end

@testset "Exceptions" begin
  @testset "Size of blocks" begin
    @test_throws Exception (DimensionMismatch, moment(data, 4, 25))
    @test_throws Exception (DimensionMismatch, cumulants(data, 3, 25))
  end
end

@testset "Cumulant helper functions" begin
  indexpart = indpart(4,2)
  @testset "indpart" begin
    @test indexpart[1].part == [[1, 2], [3, 4]]
    @test indexpart[2].part == [[1, 3], [2, 4]]
    @test indexpart[3].part == [[1, 4], [2, 3]]
  end

  @testset "operation on blocks" begin
    c2 = SymmetricTensor([1.0 2.0 3.0; 2.0 4.0 6.0; 3.0 6.0 5.0])
    blocks = accesscum((1,1,1,1), indexpart[1], c2,c2)
    @test blocks == [[1.0 2.0; 2.0 4.0], [1.0 2.0; 2.0 4.0]]
    @test accesscum((1,1,1,2), indexpart[1], c2,c2) == [[1.0 2.0; 2.0 4.0],
    [3.0 0.0; 6.0 0.0]]
    @test accesscum((1,1,1,2), indexpart[3], c2,c2) == [[3.0 0.0; 6.0 0.0],
    [1.0 2.0; 2.0 4.0]]
    block = outprodblocks(indexpart[1], blocks)
    @test block[:, :, 1, 1] == [1.0 2.0; 2.0 4.0]
    @test block[:, :, 1, 2] == [2.0 4.0; 4.0 8.0]
    @test vec((outerprodcum(4, 2, c2, c2).frame[1, 1, 1, 1])[1, 1, :, :]) == [3.0, 6.0, 6.0, 12.0]
  end

end

gaus_dat =  [[-0.88626   0.279571];
            [-0.704774  0.131896]]


@testset "Cumulants vs naive implementation" begin
  @testset "Test naive implentation" begin
    @test naivecumulant(gaus_dat, 3) ≈ zeros(Float64, 2, 2, 2)
  end
  cn = [naivecumulant(data, i) for i = 1:6]
  @testset "Square blocks" begin
    c1, c2, c3, c4, c5, c6 = cumulants(data, 6, 2)
    @test Array(c1) ≈ cn[1]
    @test Array(c2) ≈ cn[2]
    @test Array(c3) ≈ cn[3]
    @test Array(c4) ≈ cn[4]
    @test Array(c5) ≈ cn[5]
    @test Array(c6) ≈ cn[6]
  end

  @testset "Non-square blocks" begin
    c1, c2, c3, c4, c5, c6 = cumulants(data, 6, 3)
    @test Array(c1) ≈ cn[1]
    @test Array(c2) ≈ cn[2]
    @test Array(c3) ≈ cn[3]
    @test Array(c4) ≈ cn[4]
    @test Array(c5) ≈ cn[5]
    @test Array(c6) ≈ cn[6]
  end
end

@testset "test pyramid implementation" begin
  cn1, cn2, cn3, cn4, cn5, cn6, cn7, cn8 = pyramidcumulants(gaus_dat, 8)
  @test isapprox(cn1, naivecumulant(gaus_dat, 1), atol=1.0e-6)
  @test cn2 ≈ naivecumulant(gaus_dat, 2)
  @test cn3 ≈ zeros(Float64, 2, 2, 2)
  @test isapprox(cn4, zeros(Float64, 2, 2, 2, 2), atol=0.001)
  @test cn5 ≈ zeros(Float64, 2, 2, 2, 2, 2)
  @test isapprox(cn6, zeros(Float64, 2, 2, 2, 2, 2, 2), atol=0.0001)
  @test cn7 ≈ zeros(Float64, 2, 2, 2, 2, 2, 2, 2)
  @test isapprox(cn8, zeros(Float64, 2, 2, 2, 2, 2, 2, 2, 2), atol=1.0e-5)
end


@testset "Tests cumulants vs implementation from raw moments" begin
  c1, c2, c3, c4, c5, c6 = cumulants(data, 6, 2)
  cm1, cm2, cm3, cm4, cm5, cm6 = mom2cums(data, 6)
  @test cm2 ≈ Array(c2)
  @test cm3 ≈ Array(c3)
  @test cm4 ≈ Array(c4)
  @test cm5 ≈ Array(c5)
  @test cm6 ≈ Array(c6)
  llc = first_four_cumulants(data)
  @test llc[:c2] ≈ Array(c2)
  @test llc[:c3] ≈ Array(c3)
  @test llc[:c4] ≈ Array(c4)
end

cn1, cn2, cn3, cn4, cn5, cn6, cn7, cn8 = pyramidcumulants(data[:, 1:2], 8)
@testset "Cumulants vs pyramid implementation square blocks" begin
  c1, c2, c3, c4, c5, c6, c7, c8 = cumulants(data[:, 1:2], 8, 2)
  @test Array((cumulants(gaus_dat, 3))[3]) ≈ zeros(Float64, 2, 2, 2)
  @test Array(c1) ≈ cn1
  @test Array(c2) ≈ cn2
  @test Array(c3) ≈ cn3
  @test Array(c4) ≈ cn4
  @test Array(c5) ≈ cn5
  @test Array(c6) ≈ cn6
  @test Array(c7) ≈ cn7
  @test Array(c8) ≈ cn8
end

addprocs(3)
@everywhere using Cumulants
@testset "Cumulants parallel implementation" begin
  c11, c12, c13, c14, c15, c16, c17, c18 = cumulants(data[:, 1:2], 8, 2)
    @test Array(c12) ≈ cn2
    @test Array(c13) ≈ cn3
    @test Array(c14) ≈ cn4
    @test Array(c15) ≈ cn5
    @test Array(c16) ≈ cn6
    @test Array(c17) ≈ cn7
    @test Array(c18) ≈ cn8
    x = [1. 2. 3. 4. 5. 6. .7 .8 .9]
    @test Array(moment(x,1)) == [1., 2., 3., 4., 5., 6., .7, .8, .9]
end
