#!/usr/bin/env julia

using Cumulants
using PyCall
@pyimport matplotlib as mpl
mpl.use("Agg")
using PyPlot
mpl.rc("text", usetex=true)
mpl.rc("font", family="serif", size = 12)

function comptime(data::Matrix{Float64}, ccalc::Function, m::Int, b::Int)
  ccalc(data[1:4, 1:4], m, b)
  t = time_ns()
  ccalc(data, m, b)
  Float64(time_ns()-t)/1.0e9
end


function comptimesonprocs(t::Int, n::Int, m::Int, p::Int = 12)
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



function plot(t::Int, n::Int, m::Int)
  a, c = comptimesonprocs(t,n,m)
  b = a[1]./a[1:end]
  fig, ax = subplots(figsize = (4.6, 4.6))
  ax[:plot](c, b, "--x", label= "m = $m, t = $t, n = $n")
  ax[:set_ylabel]("speedup of computional time")
  ax[:set_xlabel]("core numbers")
  ax[:legend](fontsize = 12, loc = 4, ncol = 1)
  fig[:savefig]("test$n$t.eps")
end


function main()
  plot(100000, 52, 4)
end

main()
