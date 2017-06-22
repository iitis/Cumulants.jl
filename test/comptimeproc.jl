#!/usr/bin/env julia

using Cumulants
using JLD
using ArgParse



function comptime(data::Matrix{Float64}, ccalc::Function, m::Int, b::Int)
  ccalc(data[1:4, 1:4], m, b)
  t = time_ns()
  ccalc(data, m, b)
  Float64(time_ns()-t)/1.0e9
end


function comptimesonprocs(t::Int, n::Int, m::Int, p::Int)
  data = randn(t, n)
  time = Float64[]
  prc = Float64[]
  for i in 1:p
    rmprocs(procs()[2:end])
    addprocs(i-1)
    println(nprocs())
    @everywhere using Cumulants
    push!(time, comptime(data, moment, m, 3))
    push!(prc, 1.*nprocs())
  end
  time, prc
end


function savect(t::Int, n::Int, m::Int, maxprocs::Int)
  comptimes = zeros(maxprocs)
  prcs = zeros(Int, maxprocs)
  comptimes, prcs = comptimesonprocs(t,n,m,maxprocs)
  onec = copy(comptimes)
  for i in 2:maxprocs
    onec[i] = onec[1]
  end
  filename = replace("res2/"*string(m)*"_"*string(t)*"_"*string(n)*"_nprocs.jld", "[", "")
  filename = replace(filename, "]", "")
  compt = Dict{String, Any}("cumulants1c"=> onec, "cumulantsnc"=> comptimes)
  push!(compt, "t" => [t])
  push!(compt, "n" => n)
  push!(compt, "m" => m)
  push!(compt, "x" => "procs")
  push!(compt, "procs" => prcs)
  push!(compt, "functions" => [["cumulants1c", "cumulantsnc"]])
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
        default = 40
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
    end
  parsed_args = parse_args(s)
  m = parsed_args["order"]
  n = parsed_args["nvar"]
  t = parsed_args["dats"]
  p = parsed_args["maxprocs"]
  savect(t::Int, n::Int, m::Int, p::Int)
end

main(ARGS)
