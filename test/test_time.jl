if false
  addprocs()
  @everywhere using Cumulants
  rmprocs()
else
  using Cumulants
end

using Distributions
using Combinatorics
import SymmetricTensors: indices
include("test_helpers/s_naive.jl")
include("test_helpers/naivecum.jl")

"""Test a time of cumulant's calculations agains
the naive algorithm and semi naive.

Input: n_max -  maximal cumulant's order, data size t,m:
"""

function test_time(n_max::Int = 4, t::Int = 10000, m::Int = 18, naiv::Bool = true,
  sn::Bool = false)
    s = 3
    #s = 9
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
