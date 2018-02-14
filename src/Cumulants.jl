module Cumulants
  using SymmetricTensors
  using NullableArrays
  using Combinatorics
  import SymmetricTensors: indices, ind2range, sizetest, getblockunsafe
  import Distributions: moment

  #calculates moments and cumulants using block structures (SymmetricTensors)
  include("cumulant.jl")

  #naive implementation
  include("naivecumulants")

  export moment, cumulants, naivecumulant, naivemoment
end
