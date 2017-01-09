#--- seminaive formula uses partitions and reccurence, but does not use blocks

"""

  momentel(data::Matrix{T}, multind::Tuple)

Returns number, the single element of moment's tensor.
"""
momentel{T <: AbstractFloat}(data::Matrix{T}, multind::Tuple) =
  mean(mapreduce(i -> data[:,multind[i]], .*, 1:length(multind)))

"""

  indpart(n::Int)

  Returns vector of, that includes all partitions of set [1, 2, ..., n]
  into subests of size > 1 and < n

"""

function part(n::Int)
    parts = Vector{Vector{Int}}[]
    for part in partitions(1:n)
      subsetslen = map(length, part)
      if !((1 in subsetslen) | (n in subsetslen))
        @inbounds push!(parts, part)
      end
    end
    return parts
end
"""

    moment_n(data::Matrix, order::Int)

Returns n'th moment in form of order dims array
"""
function moment_n{T<:AbstractFloat}(data::Matrix{T}, order::Int)
    m = size(data,2)
    ret = zeros(fill(m, order)...)
    for ind in indices(order, m)
      @inbounds temp = momentel(data, ind)
      for per in collect(permutations([ind...]))
        @inbounds ret[per...] = temp
      end
    end
    ret
end

"""
    calculate_el(cum::Vector{Array{T}}, mulind::Tuple,
      parts::Vector{Vector{Vector{Int}}})

Returns number, element of the sum of products of lower cumulants at given
 multi-index and given ste of its partitions
"""
function calculate_el{T<:AbstractFloat}(cum::Vector{Array{T}}, mulind::Tuple,
  parts::Vector{Vector{Vector{Int}}})
    sum = 0.
    for k = 1:length(parts)
        prod = 1.
        for el in parts[k]
            @inbounds prod*= cum[size(el,1)-1][mulind[el]...]
        end
        sum += prod
    end
    sum
end

"""

  produ(cumulants::Vector{Array}, order::Int)

Returns order dims Array, the sum of products of lower cumulants
"""
function produ{T<:AbstractFloat}(cumulants::Vector{Array{T}}, order::Int)
    dats = size(cumulants[1], 2)
    sumofprod = zeros(fill(dats, order)...)
    parts = part(order)
    for ind in indices(order, dats)
      pyramid = calculate_el(cumulants, ind, parts)
      for per in collect(permutations([ind...]))
        @inbounds sumofprod[per...] = pyramid
      end
    end
    sumofprod
end

"""

  snaivecumulant(data::Matrix, maxord::Int)

Returns vector of arrays of cumulants of dims / order 2 - maxord

```
julia> gaus_dat =  [[-0.88626   0.279571];[-0.704774  0.131896]];

julia> snaivecumulant(gaus_dat, 3)[2]
2×2×2 Array{Float64,3}:
[:, :, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2] =
 0.0  0.0
 0.0  0.0
```
"""
function snaivecumulant{T<:AbstractFloat}(data::Matrix{T}, maxord::Int)
    data = data .- mean(data, 1)
    cumulants = [Base.covm(data, 0, 1, false), moment_n(data, 3)]
    for order in 4:maxord
      cumulant = moment_n(data, order) - produ(cumulants, order)
      push!(cumulants, cumulant)
    end
    cumulants
end
