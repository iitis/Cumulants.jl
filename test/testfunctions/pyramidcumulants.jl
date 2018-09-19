#--- seminaive formula uses partitions and reccurence, but does not use blocks

"""

  part(n::Int)

Returns Vector{Vector{Vector}} that includes all partitions of set [1, 2, ..., m]
into subests of size > 1 and < m

```jldoctest
julia> M =  [[-0.88626   0.279571];[-0.704774  0.131896]];

julia> part(4)
3-element Array{Array{Array{Int64,1},1},1}:
 Array{Int64,1}[[1,2],[3,4]]
 Array{Int64,1}[[1,3],[2,4]]
 Array{Int64,1}[[1,4],[2,3]]
```
"""
function part(m::Int)
    parts = Vector{Vector{Int}}[]
    for part in partitions(1:m)
      subsetslen = map(length, part)
      if !((1 in subsetslen) | (m in subsetslen))
        @inbounds push!(parts, part)
      end
    end
    parts
end

"""
    pyramidmoment(data::Matrix, m::Int)

Returns Array{Float, m}, the m'th moment tensor

```jldoctest
julia> M =  [[-0.88626   0.279571];[-0.704774  0.131896]];

julia> pyramidmoment(M, 3)
2×2×2 Array{Float64,3}:
[:, :, 1] =
 -0.523092   0.142552
  0.142552  -0.0407653

[:, :, 2] =
  0.142552   -0.0407653
 -0.0407653   0.0120729
```
"""
function pyramidmoment(data::Matrix{T}, m::Int) where T<: AbstractFloat
    n = size(data,2)
    ret = zeros(T, fill(n, m)...)
    for ind in pyramidindices(m, n)
      @inbounds temp = momel(data, ind)
      for per in collect(permutations([ind...]))
        @inbounds ret[per...] = temp
      end
    end
    ret
end

"""
    mixedel(cum::Vector{Array{T}}, mulind::Tuple, parts::Vector{Vector{Vector{Int}}})

Returns Float, element of the sum of products of lower cumulants at given
 multi-index and set of its partitions
"""
function mixedel(cum::Vector{Array{T}}, mulind::Tuple,
                  parts::Vector{Vector{Vector{Int}}}) where T<: AbstractFloat
    sum = 0.
    for k = 1:length(parts)
        prod = 1.
        for el in parts[k]
            @inbounds prod*= cum[size(el,1)][mulind[el]...]
        end
        sum += prod
    end
    sum
end

"""

  mixedarr(cumulants::Vector{Array}, m::Int)

Returns Array{Float, m}, the mixed array (sum of products of lower cumulants)
"""
function mixedarr(cumulants::Vector{Array{T}}, m::Int) where T<: AbstractFloat
    n = size(cumulants[1], 1)
    sumofprod = zeros(fill(n, m)...)
    parts = part(m)
    for ind in pyramidindices(m, n)
      pyramid = mixedel(cumulants, ind, parts)
      for per in collect(permutations([ind...]))
        @inbounds sumofprod[per...] = pyramid
      end
    end
    sumofprod
end

"""

  pyramidcumulants(X::Matrix{Float}, m::Int)

Returns [Array{Float, 2}, ..., Array{Float, m}], vector co cumulants tensors of order
2, .., m

```
julia> M =  [[-0.88626   0.279571];[-0.704774  0.131896]];

julia> pyramidcumulants(M, 3)[2]
2×2×2 Array{Float64,3}:
[:, :, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2] =
 0.0  0.0
 0.0  0.0
```
"""
function pyramidcumulants(X::Matrix{T}, m::Int = 4) where T<: AbstractFloat
    cumulants = Array{T}[]
    push!(cumulants, pyramidmoment(X, 1))
    X = X .- mean(X, dims=1)
    for i in 2:m
      if i < 4
        push!(cumulants, pyramidmoment(X, i))
      else
        cumulant = pyramidmoment(X, i) - mixedarr(cumulants, i)
        push!(cumulants, cumulant)
      end
    end
    cumulants
end
