using PyCall
@pyimport matplotlib as mpl
mpl.use("Agg")
using PyPlot
using JLD

function singleplot(filename::String, name::String, compare::String = "")
    d = load("res2/$filename"*".jld")
  if compare == ""
    comptimes = d[name]
    ylab = "computional time [s]"
  else
    comptimes = d[name]./d[compare]
    ylab = "speedup"
  end
  x = d["x"]
  t = d["t"]
  m = d["m"]
  mpl.rc("text", usetex=true)
  mpl.rc("font", family="serif", size = 8)
  fig, ax = subplots(figsize = (3, 2.3))
  for i in 1:size(comptimes, 2)
    tt = t[i]
    ax[:plot](d[x], comptimes[:,i], "--x", label= "t = $tt")
  end
  PyPlot.ylabel(ylab, labelpad = -1)
  PyPlot.xlabel(x, labelpad = -3)
  ax[:legend](fontsize = 7, loc = 2, ncol = 1)
  fig[:savefig]("res2/"*name*filename*".eps")
end


"""
  pltspeedup(comptimes::Array{Float}, m::Int, n::Vector{Int}, T::Vector{Int}, label::String)

Returns a figure in .eps format of the computional speedup of cumulants function

"""

function pltspeedup(filename::String)
  d = load("res2/$filename"*".jld")
  for f in d["functions"]
    singleplot(filename::String, f...)
  end
end

pltspeedup("4_10000_20,24_28")
