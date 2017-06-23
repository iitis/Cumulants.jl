#!/usr/bin/env julia

using NPZ
using Cumulants
addprocs()
@everywhere using Cumulants
using PyCall
@pyimport matplotlib as mpl
mpl.use("Agg")
using PyPlot
mpl.rc("text", usetex=true)
mpl.rc("font", family="serif", size = 12)


"""
  pltdiag(l::Vector{String}, codd::Vector{Float64}, cev::Vector{Float64})

Plots a chart of superdiagonal elements of cumulants of odd and even order. Those
elements are in vectors codd and cev and are named by l. Theoretical values are
computed as those of t-student standard distribution with 14 degrees of freedom.
"""
function pltdiag(l::Vector{String}, codd::Vector{Float64}, cev::Vector{Float64})
  n = length(codd)
  k = parse(Int, l[2][2])
  fig, ax = subplots(figsize = (4.6, 4.6))
  ax[:plot](cev, "o", color = "blue", label = l[2], markersize=4)
  ax[:plot]([fill(tcum(14, k), n)...], "--", color = "blue", label = "theoretical "*l[2])
  ax[:plot](codd, "s", color = "red", label = l[1], markersize=4)
  ax[:plot](zeros(n), "--", color = "red", label = "theoretical "*l[1])
  ax[:set_ylabel]("superdiagonal element")
  ax[:set_xlabel]("superdiagonal element number")
  ax[:legend](fontsize = 12, loc = 5)
  fig[:savefig]("res/diagcumel"*l[2][2:end]*".eps")
end

"""
  pltall(l::Vector{String}, v::Vector{Float64}, v1::Vector{Float64})

Plots a chart of all elements of cumulants in a vectorised form v and v1 named by l.
"""

function pltall(l::Vector{String}, v::Vector{Float64}, v1::Vector{Float64})
  fig, ax = subplots(figsize = (4.6, 4.6))
  ax[:plot](v1, "o", color = "blue", label = l[2], markersize=3)
  ax[:plot](v, "s", color = "red", label = l[1], markersize=3)
  ax[:set_ylabel]("cumulant's element")
  ax[:set_xlabel]("element's number of vectorised cumulat")
  ax[:legend](fontsize = 12, loc = 4)
  fig[:savefig]("res/allcumels"*l[2][2:end]*".eps")
end

"""
  hypdiag{T <: AbstractFloat, N}(a::Array{T, N})

Returns a vector of hyperdiagonal elements of array a
"""
hypdiag{T <: AbstractFloat, N}(a::Array{T, N}) = [a[fill(i,N)...] for i in 1:size(a,1)]

function main()
x = npzread("testdata.npz")
   c = cumulants(x, 6)
   ct = [convert(Array, c[i]) for i in 1:length(c)]
   pltdiag(["c3", "c4"], hypdiag(ct[3]), hypdiag(ct[4]))
   pltdiag(["c5", "c6"], hypdiag(ct[5]), hypdiag(ct[6]))
   pltall(["c3", "c4"], vec(ct[3]), vec(ct[4]))
   pltall(["c5", "c6"], vec(ct[5]), vec(ct[6]))
end

main()
