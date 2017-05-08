# --- copula based non Gaussian data generator
srand(42)

"""

  invers_gen(x::Vector{Float64}, theta::Float64)

Returns: Vector{Float64} of data transformed using inverse of Clayton Copula
generator with parametr theta
"""
invers_gen(x::Vector{Float64}, theta::Float64) = (1 + theta.*x).^(-1/theta)

"""

  clcopulagen(t::Int, m::Int)

Returns: Matrix{Float} of size t*m - t realisations of m dimentional random var.
generated from Clayton copula with Weibull marginals
"""
function clcopulagen(t::Int, m::Int)
  theta = 1.02
  qamma_dist = Gamma(1,1/theta)
  x = rand(t)
  u = rand(t, m)
  matrix = zeros(Float64, t, m)
  for i = 1:m
    unif_ret = invers_gen(-log(u[:,i])./quantile(qamma_dist, x), theta)
    @inbounds matrix[:,i] = quantile(Weibull(1.+0.01*i,1), unif_ret)
  end
  matrix
end

# --- uses the naive method to calculate cumulants 2 - 6

"""

  momentel(v::Vector{Vector})

Returns number, the single element of moment's tensor.
"""
momentel{T <: AbstractFloat}(v::Vector{Vector{T}}) =
  mean(mapreduce(i -> v[i], .*, 1:length(v)))

"""

  cum_el(v::Vector{Vector})

Returns number - sum of moment element and mixed element.
"""
cum_el{T<:AbstractFloat}(v::Vector{Vector{T}}) = momentel(v) + mixed_el(v...)

"""

  mixed_el(v::Vector{T}...) (input of 4, 5, or 6 vectors)

Returns number, mixed element for cumulants 4-6,
 result of contraction of vectors, (permutative sum and
  multiplication of means of elementwise products)
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

  naivecumulant(data::Matrix, order)

Returns cumulant, array of dims = order

```jldoctest
julia> gaus_dat =  [[-0.88626   0.279571];[-0.704774  0.131896]];

julia> naivecumulant(gaus_dat, 3)
2×2×2 Array{Float64,3}:
[:, :, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2] =
 0.0  0.0
 0.0  0.0

```
"""
function naivecumulant{T<:AbstractFloat}(data::Matrix{T}, order::Int = 4)
  data = data .- mean(data, 1)
  dats = size(data, 2)
  cumulant = zeros(T, fill(dats, order)...)
  if order in [2,3]
    for i = 1:(dats^order)
      ind = ind2sub((fill(dats, order)...), i)
      @inbounds cumulant[ind...] = momentel(map(k -> data[:,ind[k]],1:order))
    end
  elseif order in [4,5,6]
    for i = 1:(dats^order)
      ind = ind2sub((fill(dats, order)...), i)
      @inbounds cumulant[ind...] = cum_el(map(k -> data[:,ind[k]],1:order))
    end
  end
  return cumulant
end
