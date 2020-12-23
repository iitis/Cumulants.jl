module Cumulants
  using SymmetricTensors
  using Combinatorics
  using Distributions
  using Distributed
  import SymmetricTensors: pyramidindices, ind2range, sizetest, getblockunsafe
  import Distributions: moment
  if VERSION >= v"1.3"
   using CompilerSupportLibraries_jll
 end

  #calculates moments and cumulants using block structures (SymmetricTensors)
  include("cumulant.jl")

  #naive implementation
  include("naivecumulants.jl")

  export moment, cumulants, naivecumulant, naivemoment
end
