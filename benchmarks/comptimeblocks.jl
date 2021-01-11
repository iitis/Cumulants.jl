#!/usr/bin/env julia

using Cumulants
using JLD2
using FileIO
using ArgParse


function comptime(data::Matrix{Float64}, ccalc::Function, m::Int, b::Int)
  ccalc(data[1:4, 1:4], m, 2)
  t = time_ns()
  ccalc(data, m, b)
  Float64(time_ns()-t)/1.0e9
end


function savect(t::Vector{Int}, n::Int, m::Int)
  maxb = round(Int, sqrt(n))
  comptimes = zeros(maxb, length(t))
  println("max block size = ", maxb)
  for k in 1:length(t)
    data = randn(t[k], n)
    for b in 1:maxb
      comptimes[b, k] = comptime(data, cumulants, m, b)
      println("n = ", n)
      println("bloks size = ", b)
    end
  end
  filename = replace("res/$(m)_$(t)_$(n)_nblocks.jld2", "["=>"")
  filename = replace(filename, "]"=>"")
  compt = Dict{String, Any}("cumulants"=> comptimes)
  push!(compt, "t" => t)
  push!(compt, "n" => n)
  push!(compt, "m" => m)
  push!(compt, "x" => "block size")
  push!(compt, "block size" => [collect(1:maxb)...])
  push!(compt, "functions" => [["cumulants"]])
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
        default = 48
        help = "n, numbers of marginal variables"
        arg_type = Int
      "--dats", "-t"
        help = "t, numbers of data records"
        nargs = '*'
        default = [10000, 20000]
        arg_type = Int
    end
  parsed_args = parse_args(s)
  m = parsed_args["order"]
  n = parsed_args["nvar"]
  t = parsed_args["dats"]
  savect(t::Vector{Int}, n::Int, m::Int)
end

main(ARGS)
