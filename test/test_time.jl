using Cumulants
using Distributions
#include("test_helpers/s_naive.jl")
include("test_helpers/naivecum.jl")

"""test a time of calculations agains naive"""

function test_time(n_max::Int = 4, data_l::Int = 10000, data_w::Int = 18)
    s = 3
    data = clcopulagen(data_l, data_w);
    for n in(3:n_max)
        println("n = ", n)
        @time cumulants(data, n, s);
        println("naive")
        @time naivecumulant(data, n);
    end
end
