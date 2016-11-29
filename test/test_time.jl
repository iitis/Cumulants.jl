using Cumulants
using Distributions
#include("test_helpers/s_naive.jl")
include("test_helpers/naivecum.jl")

"""test a time of cumulant's calculations agains
the naive algorithm

input maximal cumulant's order,
data paremeters: number of records (T) number of variables (M)"""

function test_time(n_max::Int = 4, t::Int = 10000, m::Int = 18, naiv::Bool = true)
    s = 3
    data = clcopulagen(t, m);
    for n in(3:n_max)
        println("n = ", n)
        @time cumulants(data, n, s);
        if naiv
          println("naive")
          @time naivecumulant(data, n);
        end
    end
end
