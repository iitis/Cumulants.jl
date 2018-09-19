#!/usr/bin/env julia

using PyCall
@pyimport matplotlib as mpl
mpl.rc("text", usetex=true)
mpl.use("Agg")
using PyPlot
using JLD2
using FileIO
using ArgParse

function singleplot(filename::String, name::String, compare::String = "")
    d = load(filename*".jld2")
  if compare == ""
    comptimes = d[name]
    ylab = "computional time [s]"
  else
    comptimes = d[name]./d[compare]
    ylab = "speedup"
  end
  x = d["x"]
  t = d["t"]
  m = d["m"]
  mpl.rc("font", family="serif", size = 7)
  fig, ax = subplots(figsize = (2.5, 2.))
  col = ["red", "blue", "black", "green", "yellow", "orange"]
  marker = [":s", ":o", ":v", ":<", ":>", ":d"]
  for i in 1:size(comptimes, 2)
    tt = t[i]
    ax[:plot](d[x], comptimes[:,i], marker[i], label= "t = $tt", color = col[i], markersize=2.5, linewidth = 1)
  end
  PyPlot.ylabel(ylab, labelpad = -1.8)
  PyPlot.xlabel(x, labelpad = -2)
  if maximum(comptimes) > 10
    f = matplotlib[:ticker][:ScalarFormatter]()
    f[:set_powerlimits]((-3, 2))
  else
    f = matplotlib[:ticker][:FormatStrFormatter]("%.1f")
  end
  ax[:yaxis][:set_major_formatter](f)
  ax[:legend](fontsize = 6, loc = 2, ncol = 1)
  subplots_adjust(bottom = 0.12,top=0.92)
  fig[:savefig](name*filename*".pdf")
end


"""
  pltspeedup(comptimes::Array{Float}, m::Int, n::Vector{Int}, T::Vector{Int}, label::String)

Returns a figure in .pdf format of the computional speedup of cumulants function

"""

function pltspeedup(filename::String)
  d = load(filename)
  filename = replace(filename, ".jld2"=>"")
  for f in d["functions"]
    singleplot(filename::String, f...)
  end
end


function main(args)
  s = ArgParseSettings("description")
  @add_arg_table s begin
    "file"
    help = "the file name"
    arg_type = String
  end
  parsed_args = parse_args(s)
  pltspeedup(parsed_args["file"])
end

main(ARGS)
