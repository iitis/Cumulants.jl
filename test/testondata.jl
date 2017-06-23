#!/usr/bin/env julia

using JLD
using Cumulants
addprocs()
@everywhere using Cumulants

function main()
  d = try load("data/datafortests.jld")
  catch
    println("please run gendata.jl")
    return ()
  end
  c = cumulants(d["data"], 6)
  save("data/cumulants.jld", Dict("cumulant" => c))
end

main()
