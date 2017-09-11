#!/usr/bin/env julia

using JLD
using Cumulants

function main()
  d = try load("data/datafortests.jld")
  catch
    println("please run gendata.jl")
    return ()
  end
  c = cumulants(d["data"], 6)
  save("data/cumulants.jld", Dict("cumulants" => c))
end

main()