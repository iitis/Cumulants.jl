"""
  raw_moment(x::Matrix{T}, p::Int = 4)

Simmilar to raw_moments_upto_p in R, does not expoloit tensor's symmetry
pyramid structures and blocks

Returns p-dims array, a moment tensor of order p of multivariate data x
"""

function raw_moment{T <: AbstractFloat}(x::Matrix{T}, p::Int = 4)
  n,m = size(x)
  if p == 1
    return mean(x, 1)[1,:]
  else
    y = zeros(T, fill(m, p)...)
    for i in 1:n
      @inbounds xi = x[i, :]
      z = xi
      for s in 2:p
        @inbounds z = reshape(kron(xi', vec(z)), fill(m, s)...)
      end
      y = y + z
    end
  end
  y/n
end

"""
  raw_moments_upto_k(data::Matrix, k::Int = 4)

  Returns vector of arrays [Array{1}, Array{2}, ..., Array{N}] being moment
  tensors of order 1, ..., k
"""
function raw_moments_upto_k{T <: AbstractFloat}(data::Matrix{T}, k::Int = 4)
  [raw_moment(data, i) for i in 1:k]
end

"""
  function cumulants_from_moments(raw::[Array{1}, Array{2}, ..., Array{N}])

Returns vector of arrays [Array{1}, Array{2}, ..., Array{N}] being cumulant
tensors of order 1, ..., k

 Uses relation between cumulants and multivariate moments from

 @book{mccullagh1987tensor,
   title={Tensor methods in statistics},
   author={McCullagh, Peter},
   volume={161},
   year={1987},
   publisher={Chapman and Hall London}
 }

e.g. c_{ijkl} = m_{ijkl} - m_{ijk} m_{l} [4] - m_{ij} m_{kl} [3] + 2 m_{ij} m_{k} m_{l}
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
    ret += dpp[beln]*mapreduce(i->raw[k[i]][ind[part[i]]...], *, 1:beln)
  end
  ret
end

"""
  cumulatsfrommoments(x::Matrix{Float}, k::Int)

Returns a vector of 2, .., k dims Arrays{Float} of cumulant tensors
"""
mom2cums{T <: AbstractFloat}(x::Matrix{T}, k::Int) =
cumulants_from_moments(raw_moments_upto_k(x, k))[2:end]
