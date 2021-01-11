#!/usr/bin/env julia

using Distributions
using FileIO
using JLD2
using Random
using SpecialFunctions

"""
  gendat(nu::Int, t::Int)

Returns Matrix{Float64} - t realisations from t-student multivatiate distribution
with nu degress of freedom
"""

function gendat(nu::Int, t::Int = 150000000)
  cm = [[1. 0.7 0.7 0.7];[0.7 1. 0.7 0.7]; [0.7 0.7 1. 0.7]; [0.7 0.7 0.7 1]]
  p = MvTDist(nu, [0., 0., 0., 0.],cm)
  return Array(transpose(rand(p, t)))
end


"""
  tmom(nu::Int, k::Int)

Returns Float64, the k'th moment of standard t distribution with nu degreed of
freedom
"""
tmom(nu::Int, k::Int) = gamma((k+1)/2)*gamma((nu-k)/2)*nu^(k/2)/(sqrt(pi)*gamma(nu/2))
"""
  tcum(nu::Int, k::Int)

Returns Float64, the k'th cumulant of standard t distribution with nu degreed of
freedom
"""
function tcum(nu::Int, k::Int)
  if k in (1,3,5)
    return 0.
  elseif k == 2
    return tmom(nu, 2)
  elseif k == 4
    return tmom(nu, 4) - 3*tmom(nu, 2)^2
  elseif k == 6
    return tmom(nu, 6) - 15*tmom(nu, 4)*tmom(nu, 2) + 30*tmom(nu, 2)^3
  end
end

function main()
  nu = 14
  Random.seed!(42)
  data = gendat(nu::Int)
  d = Dict{String, Any}("theoretical diag" => Float64[tcum(14, k) for k in 1:6])
  push!(d, "data" => data)
  save("data/datafortests.jld2", d)
end

main()
