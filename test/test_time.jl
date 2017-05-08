#!/usr/bin/env julia

push!(LOAD_PATH, "/home/krzysztof/Dokumenty/badania/cum_calc/Cumulants/src/")
using Cumulants1
using Distributions
using PyCall
@pyimport matplotlib as mpl
mpl.use("Agg")
using PyPlot
using NPZ
import SymmetricTensors: indices

"""
  pltspeedup(comptimes::Array{Float}, M::Int, n::Vector{Int}, T::Vector{Int}, label::String)

Returns a figure in .eps format of the computional speedup of cumulants function
"""

function pltspeedup(comptimes::Array{Float64}, M::Int, n::Vector{Int}, T::Vector{Int},
   label::String)
  mpl.rc("text", usetex=true)
  mpl.rc("font", family="serif", size = 12)
  fig, ax = subplots(figsize = (4.6, 4.6))
  for i in 1:size(comptimes, 2)
    t = T[i]
    ax[:plot](n, comptimes[:,i], "--x", label= "M = $M, T = $t")
  end
  PyPlot.title("$label")
  ax[:set_ylabel]("speedup of computional time")
  ax[:set_xlabel]("tensor size")
  ax[:legend](fontsize = 12, loc = 4, ncol = 1)
  fig[:savefig]("$label$M$T$n.eps")
end

"""
  comptime(data::Matrix{Float}, ccalcf::Function, M::Int)

Returns Float, a computional time of M'th cumulant calulation of multivariate data,
given cumulant's order M and cumulats calculation function ccalc, following
functions are availavle: cumulants, naivecumulant, mom2cums, pyramidcumulants
"""

function comptime(data::Matrix{Float64}, ccalc::Function = cumulants, M::Int = 4)
  ccalc in [cumulants, naivecumulant, mom2cums, pyramidcumulants] ||
   throw(AssertionError("function not: cumulants, naivecumulant, mom2cums, pyramidcumulants"))
  ccalc(data[1:4, 1:4], M)
  t = time_ns()
  ccalc(data, M)
  Float64(time_ns()-t)/1.0e9
end

"""
  compspeedups(f::Function, M::Int, T::Vector{Int}, n::Vector{Int})

Returns Matrix, a computional speedup of M'th cumulant calculation of multivariate data,
given cumulant's order M, number of variables n, number of data realisation T,
 and cumulats calculation function ccalc, following functions are availavle:
 naivecumulant, mom2cums, pyramidcumulants
"""

function compspeedups(ccalc::Function, M::Int, T::Vector{Int}, n::Vector{Int})
  ccalc in [naivecumulant, mom2cums, pyramidcumulants] || throw(AssertionError("function not: naivecumulant, mom2cums, pyramidcumulants"))
  compt = zeros(length(n), length(T))
  for i in 1:length(T)
    for j in 1:length(n)
      data = randn(T[i], n[j])
      compt[j,i] = comptime(data, ccalc, M)/comptime(data, cumulants, M)
    end
  end
  compt
end

"""
  plotcomptime(ccalc::Function, M::Int, T::Vector{Int}, n::Vector{Int}, cash::Bool)

Returns a figure in .eps format of the computional speedup of cumulants function
vs ccalc function, following functions are availavle: naivecumulant,
mom2cums, pyramidcumulants.

M is cumulant's order, n vector of numbers of variables, T vector of numbers of
their realisations.
"""
function plotcomptime(ccalc::Function = mom2cums, M::Int = 4,
  T::Vector{Int} = [5000], n::Vector{Int} = [28, 30], cash::Bool = false)
  ccalc in [naivecumulant, mom2cums, pyramidcumulants] || throw(AssertionError("function not: naivecumulant, mom2cums, pyramidcumulants"))
  filename = string(ccalc)*string(M)*string(T)*string(n)*".npz"
  if isfile(filename)*cash
    compt = npzread(filename)
  else
    compt = compspeedups(ccalc, M, T, n)
    if cash
      npzwrite(filename, compt)
    end
  end
  pltspeedup(compt, M, n, T, string(ccalc))
end


function main()
  plotcomptime(mom2cums, 4)
  plotcomptime(naivecumulant, 4)
end

main()
