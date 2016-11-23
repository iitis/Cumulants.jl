module Cumulants
  using SymmetricTensors
  using NullableArrays
  using Iterators
  using Combinatorics
  import SymmetricTensors: indices, seg

  #calculates moments and cumulants
  include("cumulants.jl")

  #partitions. Knuth modified algorithm
  #include("/home/krzysztof/Dokumenty/badania/tensors_sym/cum_calc/text/tensor-network-project/src/part.jl")

  export momentbs, center, cumulants
end
