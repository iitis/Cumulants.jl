#!/usr/bin/env julia

using Cumulants
using JLD2
using FileIO
using ArgParse
#import Cumulants: mom2cums

include("testfunctions/leeuw_cumulants_no_nested_func.jl")

"""
  comptime(data::Matrix{Float}, f::Function, m::Int)

Returns Float, a computional time of m'th statisitics calulation of multivariate data
"""

function comptime(data::Matrix{Float64}, f::Function, m::Int)
  f(data[1:4, 1:4], m)
  t = time_ns()
  f(data, m)
  Float64(time_ns()-t)/1.0e9
end

"""
  comtimes(m::Int, t::Vector{Int}, n::Vector{Int}, f::Function)

Returns Matrix, a computional time of m'th statisitics of multivariate data,
given statisitcs's order m, number of variables n, number of data realisation t
"""

function comtimes(m::Int, t::Vector{Int}, n::Vector{Int}, f::Function)
  compt = zeros(length(n), length(t))
  for i in 1:length(t)
    for j in 1:length(n)
      data = randn(t[i], n[j])
      println("n = ", n[j])
      println("t = ", t[i])
      compt[j,i] = comptime(data, f, m)
    end
  end
  compt
end

"""
  savecomptime(m::Int, T::Vector{Int}, n::Vector{Int}, cache::Bool)

Save a file in jld2 format of the computional times of moment, naivemoment, rawmoment

"""
function savecomptime(m::Int, t::Vector{Int}, n::Vector{Int})
  filename = replace("res/"*string(m)*string(t)*string(n)*"leeuw_cums.jld2", "["=>"_")
  filename = replace(filename, "]"=>"")
  fs = [cumulants_upto_p, cumulants]
  compt = Dict{String, Any}()
  for f in fs
    fname = "$(f)"[11:end]
    println(fname)
    println("called function " , fname)
    push!(compt, fname => comtimes(m, t, n, f))
  end
  push!(compt, "t" => t)
  push!(compt, "n" => n)
  push!(compt, "m" => m)
  push!(compt, "x" => "n")
  push!(compt, "functions" => [["upto_p", "cumulants"]])
  save(filename, compt)
end

"""
  main(args)

Returns file of the speedup of momant, naivemoment rawmoment, ....
Takes optional arguments from bash
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
        default = [20, 24, 28]
        help = "n, numbers of marginal variables"
        arg_type = Int
      "--dats", "-t"
        help = "t, numbers of data records"
        nargs = '*'
        default = [10000]
        arg_type = Int
    end
  parsed_args = parse_args(s)
  m = parsed_args["order"]
  n = parsed_args["nvar"]
  t = parsed_args["dats"]
  savecomptime(m, t, n)
end

main(ARGS)
