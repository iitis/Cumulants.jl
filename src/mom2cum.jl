"""
  outer(a::Vector{Float}, b::Vector{Float})

Return Vector{Float} , vectorsed outer/kroneker product o vectors a and b
Auxiliary function for rawmoment
"""
function outer{T <: AbstractFloat}(a::Vector{T}, b::Vector{T})
    sa = size(a,1)
    sb = size(b,1)
    R = Vector{T}(sa*sb)
    m = 1
    for i = 1:sb, j = 1:sa
        @inbounds R[m] = a[j]*b[i]
        m += 1
    end
    return R
end

"""
  updvec!(A::Vector{Float}, B::Vector{Float})

Returns updated Vector{Float} A, by adding elementwisely Vector{Float} B
Auxiliary function for rawmoment 
"""
function updvec!{T<: AbstractFloat}(A::Vector{T}, B::Vector{T})
  n = size(A, 1)
  for i=1:n
    @inbounds A[i] += B[i]
  end
  return A
end

"""
  rawmoment(X::Matrix{T}, m::Int = 4)

Simmilar to raw_moments_upto_p in R, does not expoloit tensor's symmetry
pyramid structures and blocks

Returns Array{Float, m}, the m'th moment's tensor
"""

function rawmoment{T <: AbstractFloat}(X::Matrix{T}, m::Int = 4)
  t,n = size(X)
  if m == 1
    return mean(X, 1)[1,:]
  else
    y = zeros(T, n^m)
    z = T[1.]
    for i in 1:t
      for j in 1:m
        z = outer(X[i, :], z)
      end
      updvec!(y, z)
      z = T[1.]
    end
  end
  reshape(y/t, fill(n, m)...)
end

"""
  raw_moments_upto_k(X::Matrix, k::Int = 4)

Returns [Array{Float, 1}, ..., Array{Float, k}] noncentral moment tensors of
order 1, ..., k
"""
raw_moments_upto_k{T <: AbstractFloat}(X::Matrix{T}, k::Int = 4) =
  [rawmoment(X, i) for i in 1:k]

"""
  cumulants_from_moments(raw::Vector{Array{Float, i}, m = 1:k})

Returns [Array{Float, 1}, ..., Array{Float, k}] cumulant tensors of order 1, ..., k

Uses relation between cumulants and multivariate moments from e.g.
c_{ijkl} = m_{ijkl} - m_{ijk} m_{l} [4] - m_{ij} m_{kl} [3] + 2 m_{ij} m_{k} m_{l}
-6 m_{i} m_{j} m_{k} m_{l}
"""

function cumulants_from_moments{T <: AbstractFloat}(raw::Vector{Array{T}})
  k = length(raw)
  cumarr = Array(Array{Float64}, k)
  for j in 1:k
    dimr = size(raw[j])
    cumu = zeros(Float64, dimr)
    ldim = length(dimr)
    spp = collect(partitions(1:ldim))
    qpp = [(-1)^i*factorial(i) for i in 0:(ldim-1)]
    sppl = [map(length, spp[i]) for i in 1:length(spp)]
    for i in 1:prod(dimr)
      @inbounds ind = ind2sub(dimr, i)
      @inbounds cumu[ind...] = onecumulant(ind, raw, spp, sppl, qpp)
    end
    cumarr[j] = cumu
  end
  cumarr
end

"""
  onecumulant(ind::Tuple, raw::Vector{Array}, spp::Vector, sppl::Vector{Vector}, dpp::Vector)

raw - vector of moment's tensors, spp - vector of partitions, sppl - vector of sizes of partitions
dpp - vector of a factor for each product of moments (a factorial factor).

Returns Array{Float, n} the n'th cumulant tensor

"""
function onecumulant{T <: AbstractFloat}(ind::Tuple, raw::Vector{Array{T}},
  spp::Vector, sppl::Vector{Vector{Int}}, dpp::Vector{Int})
  ret = zero(T)
  for i in 1:length(spp)
    part = spp[i]
    beln = length(part)
    k = sppl[i]
    temp = one(T)
    for r in 1:beln
      temp *= raw[k[r]][ind[part[r]]...]
    end
    ret += dpp[beln]*temp
  end
  ret
end


function onecumulant11{T <: AbstractFloat}(ind::Tuple, raw::Vector{Array{T}},
  spp::Vector, sppl::Vector{Vector{Int}}, dpp::Vector{Int})
  ret = zero(T)
  for i in 1:length(spp)
    part = spp[i]
    beln = length(part)
    k = sppl[i]
    ret += dpp[beln]*mapreduce(i->raw[k[i]][ind[part[i]]...], *, 1:beln)
  end
  ret
end

"""
  cumulatsfrommoments(x::Matrix{Float}, k::Int)

Returns a vector of 1,2, .., k dims Arrays{Float} of cumulant tensors
"""
mom2cums{T <: AbstractFloat}(x::Matrix{T}, k::Int) =
  cumulants_from_moments(raw_moments_upto_k(x, k))
