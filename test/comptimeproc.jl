#!/usr/bin/env julia

using Cumulants
using JLD2
using FileIO
using ArgParse
using Distributed



function comptime(data::Matrix{Float64}, ccalc::Function, m::Int, b::Int)
  ccalc(data[1:4, 1:4], m, b)
  t = time_ns()
  ccalc(data, m, b)
  Float64(time_ns()-t)/1.0e9
end


function comptimesonprocs(t::Int, n::Int, m::Int, p::Int, b::Int)
  data = randn(t, n)
  times = zeros(p)
  for i in 1:p
    addprocs(i)
    println("number of workers = ", nworkers())
    eval(Expr(:toplevel, :(@everywhere using Cumulants)))
    times[i] = comptime(data, moment, m, b)
    rmprocs(workers())
  end
  times
end


function savect(t::Int, n::Int, m::Int, maxprocs::Int, b::Int)
  comptimes = zeros(maxprocs)
  comptimes = comptimesonprocs(t,n,m,maxprocs, b)
  onec = fill(comptimes[1], maxprocs)
  filename = replace("res/$(m)_$(t)_$(n)_$(b)_nprocs.jld2", "["=>"")
  filename = replace(filename, "]"=>"")
  compt = Dict{String, Any}("cumulants"=> onec, "cumulantsnc"=> comptimes)
  push!(compt, "t" => [t])
  push!(compt, "n" => n)
  push!(compt, "m" => m)
  push!(compt, "x" => "procs")
  push!(compt, "procs" => collect(1:maxprocs))
  push!(compt, "functions" => [["cumulants", "cumulantsnc"]])
  save(filename, compt)
end


function main(args)
  s = ArgParseSettings("description")
  @add_arg_table s begin
      "--order", "-m"
        help = "m, the order of cumulant, ndims of cumulant's tensor"
        default = 4
        arg_type = Int
      "--nvar", "-n"
        default = 50
        help = "n, numbers of marginal variables"
        arg_type = Int
      "--dats", "-t"
        help = "t, numbers of data records"
        #nargs = '*'
        default = 100000
        arg_type = Int
      "--maxprocs", "-p"
        help = "maximal number of procs"
        default = 4
        arg_type = Int
        "--blocksize", "-b"
        help = "set a block size"
        default = 2
        arg_type = Int
    end
  parsed_args = parse_args(s)
  m = parsed_args["order"]
  n = parsed_args["nvar"]
  t = parsed_args["dats"]
  p = parsed_args["maxprocs"]
  b = parsed_args["blocksize"]
  savect(t::Int, n::Int, m::Int, p::Int, b::Int)
end

main(ARGS)
