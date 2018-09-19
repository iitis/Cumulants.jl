# --- calculates moment's tensor
"""

  momel(X::Matrix{Float}, ind::Tuple)

Returns Float, an element of moment's tensor at ind multiindex

```jldoctest
julia> M =  [[-0.88626   0.279571];[-0.704774  0.131896]];

julia> momel(M, (1,1,1,1))
0.4318298020613279

```
"""
@inline momel(X::Matrix{T}, multind::Tuple) where T<: AbstractFloat = blockel(X, multind, multind, 0)

"""
  naivemoment(data::Matrix{Float}, m::Int)

Returns Array{Float, m} the m'th moment tensor

```jldoctest
julia> M =  [[-0.88626   0.279571];[-0.704774  0.131896]];

julia> naivemoment(M, 3)
2×2×2 Array{Float64,3}:
[:, :, 1] =
  -0.523092   0.142552
  0.142552  -0.0407653

[:, :, 2] =
  0.142552   -0.0407653
  -0.0407653   0.0120729

```
"""
function naivemoment(X::Matrix{T}, m::Int = 4) where T<: AbstractFloat
  n = size(X, 2)
  moment = zeros(T, fill(n, m)...,)
  for i = 1:(n^m)
    ind = Tuple(CartesianIndices((fill(n, m)...,))[i])
    @inbounds moment[ind...] = momel(X, ind)
  end
  moment
end

# --- uses the naive method to calculate cumulants 2 - 6
"""

  mixel(X::Matrix{T}, ind::Tuple)

Returns Float, mixed element for cumulants 4-6 at ind multi-index

```jldoctest
julia> M =  [[-0.88626   0.279571];[-0.704774  0.131896]];

julia> mixel(M, (1,1,1,1))
-1.232956812564408

mixel(M, (1,1,1,1,1,1))
1.015431116914347
```
"""
function mixel(X::Matrix{T}, i::Tuple) where T<: AbstractFloat
  a = zero(T)
  if length(i) == 4
    a -= momel(X, (i[1],i[2]))*momel(X, (i[3],i[4]))
    a -= momel(X, (i[1],i[3]))*momel(X, (i[2],i[4])) + momel(X, (i[1],i[4]))*momel(X, (i[2],i[3]))
  elseif length(i) == 5
    a -= momel(X, (i[1],i[2],i[3]))*momel(X, (i[4],i[5])) + momel(X, (i[1],i[2],i[4]))*momel(X, (i[3],i[5]))
    a -= momel(X, (i[1],i[2],i[5]))*momel(X, (i[3],i[4])) + momel(X, (i[2],i[3],i[4]))*momel(X, (i[1],i[5]))
    a -= momel(X, (i[2],i[3],i[5]))*momel(X, (i[1],i[4])) + momel(X, (i[1],i[3],i[4]))*momel(X, (i[2],i[5]))
    a -= momel(X, (i[1],i[3],i[5]))*momel(X, (i[2],i[4])) + momel(X, (i[3],i[4],i[5]))*momel(X, (i[1],i[2]))
    a -= momel(X, (i[2],i[4],i[5]))*momel(X, (i[1],i[3])) + momel(X, (i[1],i[4],i[5]))*momel(X, (i[2],i[3]))
  elseif length(i) == 6
    a1 = -momel(X, (i[1],i[2],i[3]))*momel(X, (i[4],i[5], i[6])) - momel(X, (i[1],i[2],i[4]))*momel(X, (i[3],i[5], i[6]))
    a1 -= momel(X, (i[1],i[2],i[5]))*momel(X, (i[3],i[4], i[6])) + momel(X, (i[1],i[2],i[6]))*momel(X, (i[3],i[4], i[5]))
    a1 -= momel(X, (i[1],i[3],i[4]))*momel(X, (i[2],i[5], i[6])) + momel(X, (i[1],i[3],i[5]))*momel(X, (i[2],i[4], i[6]))
    a1 -= momel(X, (i[1],i[3],i[6]))*momel(X, (i[2],i[4], i[5])) + momel(X, (i[1],i[4],i[5]))*momel(X, (i[2],i[3], i[6]))
    a1 -= momel(X, (i[1],i[4],i[6]))*momel(X, (i[2],i[3], i[5])) + momel(X, (i[1],i[5],i[6]))*momel(X, (i[2],i[3], i[4]))
    a2 = -momel(X, (i[1],i[2],i[3],i[4]))*momel(X, (i[5], i[6])) - momel(X, (i[1],i[2],i[3],i[5]))*momel(X, (i[4],i[6]))
    a2 -= momel(X, (i[1],i[2],i[3],i[6]))*momel(X, (i[4], i[5])) + momel(X, (i[1],i[2],i[4],i[5]))*momel(X, (i[3], i[6]))
    a2 -= momel(X, (i[1],i[2],i[4],i[6]))*momel(X, (i[3], i[5])) + momel(X, (i[1],i[2],i[5],i[6]))*momel(X, (i[3], i[4]))
    a2 -= momel(X, (i[3],i[4],i[5],i[6]))*momel(X, (i[1], i[2])) + momel(X, (i[1],i[3],i[4],i[5]))*momel(X, (i[2], i[6]))
    a2 -= momel(X, (i[1],i[3],i[4],i[6]))*momel(X, (i[2], i[5])) + momel(X, (i[1],i[3],i[5],i[6]))*momel(X, (i[2], i[4]))
    a2 -= momel(X, (i[2],i[4],i[5],i[6]))*momel(X, (i[1], i[3])) + momel(X, (i[1],i[4],i[5],i[6]))*momel(X, (i[2], i[3]))
    a2 -= momel(X, (i[2],i[3],i[5],i[6]))*momel(X, (i[1], i[4])) + momel(X, (i[2],i[3],i[4],i[6]))*momel(X, (i[1], i[5]))
    a2 -= momel(X, (i[2],i[3],i[4],i[5]))*momel(X, (i[1], i[6]))
    a3 = -momel(X, (i[1],i[2]))*momel(X, (i[3],i[4]))*momel(X, (i[5], i[6]))
    a3 -= momel(X, (i[1],i[2]))*momel(X, (i[3],i[5]))*momel(X, (i[4], i[6]))
    a3 -= momel(X, (i[1],i[2]))*momel(X, (i[3],i[6]))*momel(X, (i[4], i[5]))
    a3 -= momel(X, (i[1],i[3]))*momel(X, (i[2],i[4]))*momel(X, (i[5], i[6]))
    a3 -= momel(X, (i[1],i[3]))*momel(X, (i[2],i[5]))*momel(X, (i[4], i[6]))
    a3 -= momel(X, (i[1],i[3]))*momel(X, (i[2],i[6]))*momel(X, (i[4], i[5]))
    a3 -= momel(X, (i[1],i[4]))*momel(X, (i[2],i[3]))*momel(X, (i[5], i[6]))
    a3 -= momel(X, (i[1],i[5]))*momel(X, (i[2],i[3]))*momel(X, (i[4], i[6]))
    a3 -= momel(X, (i[1],i[6]))*momel(X, (i[2],i[3]))*momel(X, (i[4], i[5]))
    a3 -= momel(X, (i[1],i[4]))*momel(X, (i[2],i[5]))*momel(X, (i[3], i[6]))
    a3 -= momel(X, (i[1],i[4]))*momel(X, (i[2],i[6]))*momel(X, (i[3], i[5]))
    a3 -= momel(X, (i[1],i[5]))*momel(X, (i[2],i[4]))*momel(X, (i[3], i[6]))
    a3 -= momel(X, (i[1],i[6]))*momel(X, (i[2],i[4]))*momel(X, (i[3], i[5]))
    a3 -= momel(X, (i[1],i[5]))*momel(X, (i[2],i[6]))*momel(X, (i[3], i[4]))
    a3 -= momel(X, (i[1],i[6]))*momel(X, (i[2],i[5]))*momel(X, (i[3], i[4]))
    a += a1+a2-2*a3
  end
  a
end

"""
  naivecumulant(data::Matrix, m::Int)

Returns Array{Float, m} the m'th cumulant tensor

```jldoctest
julia> M =  [[-0.88626   0.279571];[-0.704774  0.131896]];

julia> naivecumulant(M, 3)
2×2×2 Array{Float64,3}:
[:, :, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2] =
 0.0  0.0
 0.0  0.0

```
"""
function naivecumulant(X::Matrix{T}, m::Int = 4) where T<: AbstractFloat
  m < 7 || throw(AssertionError("naive implementation of $m cumulant not supported"))
  if m == 1
    return naivemoment(X,m)
  end
  X = X .- mean(X, dims=1)
  ret = naivemoment(X,m)
  if m in [4,5,6]
    n = size(X, 2)
    for i = 1:(n^m)
      ind = Tuple(CartesianIndices((fill(n, m)...,))[i])
      @inbounds ret[ind...] += mixel(X, ind)
    end
  end
    return ret
end
