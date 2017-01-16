if false
  addprocs()
  @everywhere using Cumulants
  rmprocs()
else
  using Cumulants
end
using Distributions
import SymmetricTensors: indices

include("test_helpers/s_naive.jl")
include("test_helpers/naivecum.jl")

"""

Test a time of cumulant's calculations agains the naive algorithm and semi naive.
"""
function test_time(maxorder::Int = 4, realisations::Int = 1000, nvar::Int = 30,
  compwnaiv::Bool = false, compwseminaiv::Bool = false, bls::Int = 2)
    data = clcopulagen(realisations, nvar);
    for order in(3:maxorder)
        println("n = ", order)
        @time cumulants(data, order, bls);
        if compwnaiv
          println("naive")
          @time naivecumulant(data, order);
        end
        if compwseminaiv
          println("seminaive")
          @time snaivecumulant(data, order);
        end
    end
end
