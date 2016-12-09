if false
  addprocs()
  @everywhere using Cumulants
  rmprocs()
else
  using Cumulants
end
using Distributions
import SymmetricTensors: indices
import Cumulants: momentel

include("test_helpers/s_naive.jl")
include("test_helpers/naivecum.jl")

"""Test a time of cumulant's calculations agains
the naive algorithm and semi naive.

Input: n_max -  maximal cumulant's order, t,m - test data size, naiv, sn - use
(naive, semi naive algorithm for time comparison), s - number of blocks 
"""

function test_time(n_max::Int = 4, t::Int = 1000, m::Int = 30, naiv::Bool = true,
  sn::Bool = false, s::Int = 3)
    data = clcopulagen(t, m);
    for n in(3:n_max)
        println("n = ", n)
        @time cumulants(data, n, s);
        if naiv
          println("naive")
          @time naivecumulant(data, n);
        end
        if sn
          println("seminaive")
          @time snaivecumulant(data, n);
        end
    end
end
