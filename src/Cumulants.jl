module Cumulants
  using SymmetricTensors
  using NullableArrays
  using Iterators
  using Combinatorics
  import SymmetricTensors: indices, ind2range, sizetest
  import Distributions: moment

  #calculates moments and cumulants
  include("cumulants.jl")

  export moment, cumulants
end
