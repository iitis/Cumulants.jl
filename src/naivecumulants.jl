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
function momel{T <: AbstractFloat}(data::Matrix{T}, multind::Tuple)
  ret = 0.
  t = size(data, 1)
  for l in 1:t
    temp = 1.
    for i in multind
      @inbounds temp *= data[l,i]
    end
    ret += temp
  end
  ret/t
end


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
function naivemoment{T<:AbstractFloat}(X::Matrix{T}, m::Int = 4)
  n = size(X, 2)
  moment = zeros(T, fill(n, m)...)
  for i = 1:(n^m)
    ind = ind2sub((fill(n, m)...), i)
    @inbounds moment[ind...] = momel(X, ind)
  end
  moment
end

# --- uses the naive method to calculate cumulants 2 - 6

"""

  cumel(X::Matrix{Float}, ind::Tuple)

Returns Float an element of cumulant's tensor at ind multiindex

```jldoctest
julia> M =  [[-0.88626   0.279571];[-0.704774  0.131896]];

julia> cumel(M, (1,1,1,1))
-0.80112701050308
```
"""
cumel{T<:AbstractFloat}(X::Matrix{T}, ind::Tuple) = momel(X, ind) + mixel(X, ind)

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

function mixel{T<:AbstractFloat}(X::Matrix{T}, ind::Tuple)
  A = X[:,ind[1]]
  B = X[:,ind[2]]
  C = X[:,ind[3]]
  D = X[:,ind[4]]
  if length(ind) == 4
    return -mean(A.*B)*mean(C.*D) - mean(A.*C)*mean(B.*D) - mean(A.*D)*mean(B.*C)
  elseif length(ind) == 5
    E = X[:,ind[5]]
    a = -mean(A.*B.*C)*mean(D.*E) - mean(A.*B.*D)*mean(C.*E) - mean(A.*B.*E)*mean(D.*C)
    a -= mean(D.*B.*C)*mean(A.*E) + mean(E.*B.*C)*mean(D.*A) + mean(A.*D.*C)*mean(B.*E)
    a -= mean(A.*E.*C)*mean(B.*D) + mean(D.*E.*C)*mean(A.*B) + mean(D.*B.*E)*mean(A.*C)
    a -= mean(A.*D.*E)*mean(C.*B)
    return a
  elseif length(ind) == 6
    E = X[:,ind[5]]
    F = X[:,ind[6]]
    a1 = -mean(A.*B.*C)*mean(D.*E.*F) - mean(A.*B.*D)*mean(C.*E.*F) - mean(A.*B.*E)*mean(C.*D.*F)
    a1 -= mean(A.*B.*F)*mean(C.*D.*E) + mean(A.*C.*D)*mean(B.*E.*F) + mean(A.*C.*E)*mean(B.*D.*F)
    a1 -= mean(A.*C.*F)*mean(B.*D.*E) + mean(A.*D.*E)*mean(B.*C.*F) + mean(A.*D.*F)*mean(B.*C.*E)
    a1 -= mean(A.*E.*F)*mean(B.*C.*D)
    a2 = -mean(A.*B.*C.*D)*mean(E.*F) - mean(A.*B.*C.*E)*mean(D.*F) - mean(A.*B.*C.*F)*mean(D.*E)
    a2 -= mean(A.*B.*D.*E)*mean(C.*F) + mean(A.*B.*D.*F)*mean(C.*E) + mean(A.*B.*E.*F)*mean(C.*D)
    a2 -= mean(A.*B)*mean(C.*D.*E.*F) + mean(A.*C.*D.*E)*mean(B.*F) + mean(A.*C.*D.*F)*mean(B.*E)
    a2 -= mean(A.*C.*E.*F)*mean(B.*D) + mean(A.*C)*mean(B.*D.*E.*F) + mean(A.*D.*E.*F)*mean(B.*C)
    a2 -= mean(A.*D)*mean(B.*C.*E.*F) + mean(A.*E)*mean(B.*C.*D.*F) + mean(A.*F)*mean(B.*C.*D.*E)
    a3 = -mean(A.*B)*mean(C.*D)*mean(E.*F) - mean(A.*B)*mean(C.*E)*mean(D.*F)
    a3 -= mean(A.*B)*mean(C.*F)*mean(D.*E) + mean(A.*C)*mean(B.*D)*mean(E.*F)
    a3 -= mean(A.*C)*mean(B.*E)*mean(D.*F) + mean(A.*C)*mean(B.*F)*mean(D.*E)
    a3 -= mean(A.*D)*mean(B.*C)*mean(E.*F) + mean(A.*E)*mean(B.*C)*mean(D.*F)
    a3 -= mean(A.*F)*mean(B.*C)*mean(D.*E) + mean(A.*D)*mean(B.*E)*mean(C.*F)
    a3 -= mean(A.*D)*mean(B.*F)*mean(C.*E) + mean(A.*E)*mean(B.*D)*mean(C.*F)
    a3 -= mean(A.*F)*mean(B.*D)*mean(C.*E) + mean(A.*E)*mean(B.*F)*mean(C.*D)
    a3 -= mean(A.*F)*mean(B.*E)*mean(C.*D)
    return a1+a2-2*a3
  end
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
function naivecumulant{T<:AbstractFloat}(X::Matrix{T}, m::Int = 4)
  X = X .- mean(X, 1)
  n = size(X, 2)
  cumulant = zeros(T, fill(n, m)...)
  if m in [2,3]
    for i = 1:(n^m)
      ind = ind2sub((fill(n, m)...), i)
      @inbounds cumulant[ind...] = momel(X, ind)
    end
  elseif m in [4,5,6]
    for i = 1:(n^m)
      ind = ind2sub((fill(n, m)...), i)
      @inbounds cumulant[ind...] = cumel(X, ind)
    end
  end
  cumulant
end
