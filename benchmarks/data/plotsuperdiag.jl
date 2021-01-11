#!/usr/bin/env julia

using JLD2
using FileIO
using SymmetricTensors
using PyCall
@pyimport matplotlib as mpl
using PyPlot
mpl.rc("text", usetex=true)
mpl.rc("font", family="serif", size = 8)


"""
  pltdiag()

Plots a chart of superdiagonal elements of cumulants of and its theoretical values
"""
function pltdiag()
  tdiag = try load("datafortests.jld2")["theoretical diag"]
  catch
    println("please run test/gandata.jl and test/testondata.jl")
  return ()
  end
  cum = try load("cumulants.jld2")["cumulants"]
  catch
    println("please run test/testondata.jl")
    return ()
  end
  n = cum[1].dats
  fig, ax = subplots(figsize = (3., 2.3))
  col = ["cyan", "brown", "green", "red", "blue", "black"]
  for order in (2,4,5,6)
    c = cum[order]
    ax[:plot](diag(c), "o", color = col[order], label = "$order cumulant", markersize=3)
    ax[:plot]([fill(tdiag[order], n)...], "--", color = col[order], label = "theoretical")
  end
  PyPlot.ylabel("superdiagonal elements", labelpad = -1)
  PyPlot.xlabel("superdiagonal number", labelpad = -3)
  ax[:legend](fontsize = 4.5, loc = 5)
  fig[:savefig]("diagcumels.pdf")
end



function main()
  pltdiag()
end

main()
