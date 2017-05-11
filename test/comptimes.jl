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
  name = replace("$label$M$T$n", "[", "_")
  name = replace(name, "]", "")
  fig[:savefig](name*".eps")
end

"""
  comptime(data::Matrix{Float}, ccalcf::Function, M::Int)

Returns Float, a computional time of M'th cumulant calulation of multivariate data,
given cumulant's order M and cumulats calculation function ccalc, following
functions are availavle: cumulants, naivecumulant, mom2cums, pyramidcumulants
"""

function comptime(data::Matrix{Float64}, ccalc::Function, M::Int)
  ccalc(data[1:4, 1:4], M)
  t = time_ns()
  ccalc(data, M)
  Float64(time_ns()-t)/1.0e9
end


function comptime(data::Matrix{Float64}, M::Int)
  cumulants(data[1:4, 1:4], M, 3)
  t = time_ns()
  cumulants(data, M, 3)
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
function plotcomptime(ccalc::Function, M::Int, T::Vector{Int}, n::Vector{Int}, cash::Bool)
  filename = replace(string(ccalc)*string(M)*string(T)*string(n)*".npz", "[", "_")
  filename = replace(filename, "]", "")
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

"""
  main(args)

Returns plots of the speedup of using cumulants function vs. naivecumulants and
mom2cums implementations. Takes optional arguments from bash
"""
function main(args)
  s = ArgParseSettings("description")
  @add_arg_table s begin
      "--order", "-M"
        help = "M, the order of cumulant, ndims of cumulant's tensor"
        default = 4
        arg_type = Int
      "--nvar", "-n"
        nargs = '*'
        default = [22, 24]
        help = "n, numbers of marginal variables"
        arg_type = Int
      "--dats", "-T"
        help = "T, numbers of data records"
        nargs = '*'
        default = [4000]
        arg_type = Int
      "--cash", "-c"
        help = "indicates if computional times should be saved in a file or read
          from a file"
        default = false
        arg_type = Bool
    end
  parsed_args = parse_args(s)
  M = parsed_args["order"]
  n = parsed_args["nvar"]
  T = parsed_args["dats"]
  cash = parsed_args["cash"]
  plotcomptime(mom2cums, M, T, n, cash)
  plotcomptime(naivecumulant, M, T, n, cash)
end

main(ARGS)
