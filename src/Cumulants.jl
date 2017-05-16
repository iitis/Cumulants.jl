module Cumulants
  using SymmetricTensors
  using NullableArrays
  using Iterators
  using Combinatorics
  using Distributions
  using TensorOperations
  import SymmetricTensors: indices, ind2range, sizetest
  import Distributions: moment

  #calculates moments and cumulants
  include("cumulants.jl")

  include("pyramidcumulants.jl")
  include("naivecumulants.jl")
  include("mom2cum.jl")

  export moment, cumulants, getcumulant, pyramidcumulants, naivecumulant, mom2cums
end
