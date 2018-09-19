#!/usr/bin/env julia

using JLD2
using FileIO
using Cumulants

function main()
  d = try load("data/datafortests.jld2")
  catch
    println("please run gendata.jl")
    return ()
  end
  c = cumulants(d["data"], 6)
  save("data/cumulants.jld2", Dict("cumulants" => c))
end

main()
