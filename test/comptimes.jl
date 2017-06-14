#!/usr/bin/env julia

using Cumulants
using Distributions
using PyCall
@pyimport matplotlib as mpl
mpl.use("Agg")
using PyPlot
using NPZ
using ArgParse
import SymmetricTensors: indices
import Cumulants: rawmoment, mom2cums
"""
  pltspeedup(comptimes::Array{Float}, m::Int, n::Vector{Int}, T::Vector{Int}, label::String)

Returns a figure in .eps format of the computional speedup of cumulants function
"""

function pltspeedup(comptimes::Array{Float64}, m::Int, n::Vector{Int}, T::Vector{Int},
   label::String)
  mpl.rc("text", usetex=true)
  mpl.rc("font", family="serif", size = 8)
  fig, ax = subplots(figsize = (3, 2.3))
  for i in 1:size(comptimes, 2)
    t = T[i]
    ax[:plot](n, comptimes[:,i], "--x", label= "t = $t")
  end
  #PyPlot.title("$label")
  #ax[:set_ylabel]("speedup of computional time")
  PyPlot.ylabel("speedup of computional time", labelpad = -3)
  PyPlot.xlabel("m", labelpad = -3)
  #ax[:set_xlabel]("tensor size - m")
  ax[:legend](fontsize = 8, loc = 2, ncol = 1)
  name = replace("$label$m$T$n", "[", "_")
  name = replace(name, "]", "")
  fig[:savefig]("res2/"*name*".eps")
end

"""
  comptime(data::Matrix{Float}, ccalcf::Function, M::Int)

Returns Float, a computional time of m'th cumulant calulation of multivariate data,
given cumulant's order m and cumulats calculation function ccalc, following
functions are availavle: cumulants, naivecumulant, mom2cums, pyramidcumulants
"""

function comptime(data::Matrix{Float64}, ccalc::Function, m::Int)
  ccalc(data[1:4, 1:4], m)
  t = time_ns()
  ccalc(data, m)
  Float64(time_ns()-t)/1.0e9
end

function comptime(data::Matrix{Float64}, ccalc::Function, m::Int, b::Int)
  ccalc(data[1:4, 1:4], m, b)
  t = time_ns()
  ccalc(data, m, b)
  Float64(time_ns()-t)/1.0e9
end

"""
  compspeedups(f::Function, M::Int, T::Vector{Int}, n::Vector{Int})

Returns Matrix, a computional speedup of m'th cumulant calculation of multivariate data,
given cumulant's order M, number of variables n, number of data realisation T,
 and cumulats calculation function ccalc, following functions are availavle:
 naivecumulant, mom2cums, pyramidcumulants
"""

function compspeedups(ccalc::Function, m::Int, T::Vector{Int}, n::Vector{Int}, f::Function)
  compt = zeros(length(n), length(T))
  for i in 1:length(T)
    for j in 1:length(n)
      data = randn(T[i], n[j])
      compt[j,i] = comptime(data, ccalc, m)/comptime(data, f, m, 3)
    end
  end
  compt
end

"""
  plotcomptime(ccalc::Function, M::Int, T::Vector{Int}, n::Vector{Int}, cache::Bool)

Returns a figure in .eps format of the computional speedup of cumulants function
vs ccalc function, following functions are availavle: naivecumulant,
mom2cums, pyramidcumulants.

M is cumulant's order, n vector of numbers of variables, T vector of numbers of
their realisations.
"""
function plotcomptime(ccalc::Function, m::Int, T::Vector{Int}, n::Vector{Int}, cache::Bool, f::Function = cumulants)
  filename = replace("res2/"*string(ccalc)*string(m)*string(T)*string(n)*".npz", "[", "_")
  filename = replace(filename, "]", "")
  if isfile(filename)*cache
    compt = npzread(filename)
  else
    compt = compspeedups(ccalc, m, T, n, f)
    if cache
      npzwrite(filename, compt)
    end
  end
  pltspeedup(compt, m, n, T, string(ccalc))
end

"""
  main(args)

Returns plots of the speedup of using cumulants function vs. naivecumulants and
mom2cums implementations. Takes optional arguments from bash
"""
function main(args)
  s = ArgParseSettings("description")
  @add_arg_table s begin
      "--order", "-m"
        help = "m, the order of cumulant, ndims of cumulant's tensor"
        default = 4
        arg_type = Int
      "--nvar", "-n"
        nargs = '*'
        default = [22, 24]
        help = "n, numbers of marginal variables"
        arg_type = Int
      "--dats", "-t"
        help = "t, numbers of data records"
        nargs = '*'
        default = [4000]
        arg_type = Int
      "--cache", "-c"
        help = "indicates if computional times should be saved in a file or read
          from a file"
        default = false
        arg_type = Bool
    end
  parsed_args = parse_args(s)
  m = parsed_args["order"]
  n = parsed_args["nvar"]
  t = parsed_args["dats"]
  cache = parsed_args["cache"]
  plotcomptime(naivemoment, m, t, n, cache, moment)
  plotcomptime(mom2cums, m, t, n, cache)
  plotcomptime(naivecumulant, m, t, n, cache)
end

main(ARGS)
