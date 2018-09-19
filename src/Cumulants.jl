module Cumulants
  using SymmetricTensors
  using Combinatorics
  using Statistics
  using Distributed
  import SymmetricTensors: pyramidindices, ind2range, sizetest, getblockunsafe
  import Distributions: moment

  #calculates moments and cumulants using block structures (SymmetricTensors)
  include("cumulant.jl")

  #naive implementation
  include("naivecumulants.jl")

  export moment1, cumulants, naivecumulant, naivemoment
end
