module Cumulants
  using SymmetricTensors
  using NullableArrays
  using Iterators
  using Combinatorics
  #using Distributions
  import SymmetricTensors: indices, ind2range, sizetest
  import Distributions: moment

  #calculates moments and cumulants using block structures (SymmetricTensors)
  include("cumulants.jl")

  #other
  include("pyramidcumulants.jl")
  include("naivecumulants.jl")
  include("mom2cum.jl")

  export moment, cumulants, getcumulant, naivecumulant, naivemoment
end
