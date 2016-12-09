# --- copula based non Gaussian data generator
srand(42)

"""
Inverse of Clayton Copula generator.

Input: x - data vector, theta - parameter.

Returns: transformed data vector.
"""
invers_gen(x::Vector{Float64}, theta::Float64) = (1 + theta.*x).^(-1/theta)

"""
Uses Clayton copula with Weibull marginals to generate multivariate data that
are not gaussian distributed for cumulantgs tests.

Input: m - dimention of multivariete data, t - number of random data relaisations

Returns: Matrix{Float} of size t*m - t realisations of m dimentional random var.
"""
function clcopulagen(t::Int, m::Int)
  theta = 1.02
  qamma_dist = Gamma(1,1/theta)
  x = rand(t)
  u = rand(t, m)
  ret = zeros(Float64, t, m)
  for i = 1:m
    @inbounds unif_ret = invers_gen(-log(u[:,i])./quantile(qamma_dist, x), theta)
    @inbounds ret[:,i] = quantile(Weibull(1.+0.01*i,1), unif_ret)
  end
  ret
end

# --- uses the naive method to calculate cumulants 2 - 6
"""
The element of the n'th cumulant tensor.

Input: n vectors of data

Output: Float - sum of n'th moment's element and mixed element.
"""
cum_el{T<:AbstractFloat}(d::Vector{Vector{T}}) = momentel(d) + mixed_el(d...)

"""
The mixed element for cumulants 4-6 respectivelly.

Input: 4-6 vectors of data,

Output: Float - result of contraction of vectors, (permutative sum and
  multiplication of means of elementwise products).
"""
function mixed_el{T<:AbstractFloat}(A::Vector{T}, B::Vector{T}, C::Vector{T}, D::Vector{T})
  -mean(A.*B)*mean(C.*D) - mean(A.*C)*mean(B.*D) - mean(A.*D)*mean(B.*C)
end
function mixed_el{T<:AbstractFloat}(A::Vector{T}, B::Vector{T}, C::Vector{T},
  D::Vector{T}, E::Vector{T})
  a = -mean(A.*B.*C)*mean(D.*E) - mean(A.*B.*D)*mean(C.*E) - mean(A.*B.*E)*mean(D.*C)
  a -= mean(D.*B.*C)*mean(A.*E) + mean(E.*B.*C)*mean(D.*A) + mean(A.*D.*C)*mean(B.*E)
  a -= mean(A.*E.*C)*mean(B.*D) + mean(D.*E.*C)*mean(A.*B) + mean(D.*B.*E)*mean(A.*C)
  a -= mean(A.*D.*E)*mean(C.*B)
  return a
end
function mixed_el{T<:AbstractFloat}(A::Vector{T}, B::Vector{T}, C::Vector{T},
  D::Vector{T}, E::Vector{T}, F::Vector{T})
  block1 = -mean(A.*B.*C)*mean(D.*E.*F) - mean(A.*B.*D)*mean(C.*E.*F) - mean(A.*B.*E)*mean(C.*D.*F)
  block1 -= mean(A.*B.*F)*mean(C.*D.*E) + mean(A.*C.*D)*mean(B.*E.*F) + mean(A.*C.*E)*mean(B.*D.*F)
  block1 -= mean(A.*C.*F)*mean(B.*D.*E) + mean(A.*D.*E)*mean(B.*C.*F) + mean(A.*D.*F)*mean(B.*C.*E)
  block1 -= mean(A.*E.*F)*mean(B.*C.*D)
  block2 = -mean(A.*B.*C.*D)*mean(E.*F) - mean(A.*B.*C.*E)*mean(D.*F) - mean(A.*B.*C.*F)*mean(D.*E)
  block2 -= mean(A.*B.*D.*E)*mean(C.*F) + mean(A.*B.*D.*F)*mean(C.*E) + mean(A.*B.*E.*F)*mean(C.*D)
  block2 -= mean(A.*B)*mean(C.*D.*E.*F) + mean(A.*C.*D.*E)*mean(B.*F) + mean(A.*C.*D.*F)*mean(B.*E)
  block2 -= mean(A.*C.*E.*F)*mean(B.*D) + mean(A.*C)*mean(B.*D.*E.*F) + mean(A.*D.*E.*F)*mean(B.*C)
  block2 -= mean(A.*D)*mean(B.*C.*E.*F) + mean(A.*E)*mean(B.*C.*D.*F) + mean(A.*F)*mean(B.*C.*D.*E)
  block3 = -mean(A.*B)*mean(C.*D)*mean(E.*F) - mean(A.*B)*mean(C.*E)*mean(D.*F)
  block3 -= mean(A.*B)*mean(C.*F)*mean(D.*E) + mean(A.*C)*mean(B.*D)*mean(E.*F)
  block3 -= mean(A.*C)*mean(B.*E)*mean(D.*F) + mean(A.*C)*mean(B.*F)*mean(D.*E)
  block3 -= mean(A.*D)*mean(B.*C)*mean(E.*F) + mean(A.*E)*mean(B.*C)*mean(D.*F)
  block3 -= mean(A.*F)*mean(B.*C)*mean(D.*E) + mean(A.*D)*mean(B.*E)*mean(C.*F)
  block3 -= mean(A.*D)*mean(B.*F)*mean(C.*E) + mean(A.*E)*mean(B.*D)*mean(C.*F)
  block3 -= mean(A.*F)*mean(B.*D)*mean(C.*E) + mean(A.*E)*mean(B.*F)*mean(C.*D)
  block3 -= mean(A.*F)*mean(B.*E)*mean(C.*D)
  return block2+block1-2*block3
end

"""
Calculates cumulant using naive algorithm.

Input: data - input data in matrix form, order - Int in [2,3,...,6], cumulant's
order.

Output: cumulant, tensor of size m ^ order
"""
function naivecumulant{T<:AbstractFloat}(data::Matrix{T}, order::Int = 4)
  data = center(data)
  m = size(data, 2)
  ret = zeros(T, fill(m, order)...)
  if order in [2,3]
    for i = 1:(m^order)
      @inbounds ind = ind2sub((fill(m, order)...), i)
      @inbounds ret[ind...] = momentel(map(k -> data[:,ind[k]],1:order))
    end
  elseif order in [4,5,6]
    for i = 1:(m^order)
      @inbounds ind = ind2sub((fill(m, order)...), i)
      @inbounds ret[ind...] = cum_el(map(k -> data[:,ind[k]],1:order))
    end
  end
  return ret
end
