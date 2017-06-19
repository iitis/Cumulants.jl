using PyCall
@pyimport matplotlib as mpl
mpl.use("Agg")
using PyPlot
using JLD


"""
  pltspeedup(comptimes::Array{Float}, m::Int, n::Vector{Int}, T::Vector{Int}, label::String)

Returns a figure in .eps format of the computional speedup of cumulants function
"""

function pltspeedup(filename::String)
  d = load("res2/$filename")
  singleplot(d, "naivemoment", "moment")
  singleplot(d, "rawmoment", "moment")
  singleplot(d, "naivecumulant", "cumulants")
  singleplot(d, "mom2cums", "cumulants")
end

function singleplot(d::Dict, name::String, compare::String)
  comptimes = d[name]./d[compare]
  t = d["t"]
  n = d["n"]
  m = d["m"]
  mpl.rc("text", usetex=true)
  mpl.rc("font", family="serif", size = 8)
  fig, ax = subplots(figsize = (3, 2.3))
  for i in 1:size(comptimes, 2)
    tt = t[i]
    ax[:plot](n, comptimes[:,i], "--x", label= "m = $m, t = $tt")
  end
  PyPlot.ylabel("speedup of computional time", labelpad = -1)
  PyPlot.xlabel("m", labelpad = -3)
  ax[:legend](fontsize = 8, loc = 2, ncol = 1)
  name = replace("$name$m$t$n", "[", "_")
  name = replace(name, "]", "")
  fig[:savefig]("res2/"*name*".eps")
end

pltspeedup("4_4000_22,24.jld")
