#--- semi naive formula uses partitions and reccurence does not use blocks
"""
Produces all set partitions of multiindex i

Output: Vector{Vector{Vector{Int}}} - vector of sets of partitions.
"""

function partitions2(i::Tuple)
    part = Vector{Vector{Int}}[]
    n = length(i)
    for p in partitions([i...])
      s = map(length, p)
      if !((1 in s) | (n in s))
        @inbounds push!(part, p)
      end
    end
    return part
end

"""
Calculates n'th moment in form of symmetric n dimestional array

Input: data - matrix,

Output: array of size m^n
"""
function moment_n{T<:AbstractFloat}(data::Matrix{T}, order::Int)
    m = size(data,2)
    ret = zeros(fill(m, order)...)
    for ind in indices(order, m)
      @inbounds temp = momentel(map(k -> data[:,ind[k]],1:order))
      for per in collect(permutations([ind...]))
        @inbounds ret[per...] = temp
      end
    end
    ret
end

"""
element of the sum of products of lower cumulants

output float number
"""
function calculate_el{T<:AbstractFloat}(c::Vector{Array{T}}, i::Tuple)
    part = partitions2(i)
    sum = 0.
    for k = 1:length(part)
        prod = 1.
        for el in part[k]
            @inbounds prod*= c[size(el,1)-1][el...]
        end
        sum += prod
    end
    sum
end

"""
the sum of products of lower cumulants

output array
"""
function produ{T<:AbstractFloat}(c::Vector{Array{T}}, n::Int)
    m = size(c[1], 2)
    ret = zeros(fill(m,n)...)
    for ind in indices(n, m)
      @inbounds temp = calculate_el(c, ind)
      for per in collect(permutations([ind...]))
        @inbounds ret[per...] = temp
      end
    end
    ret
end

"""
calculate cumulants

imput: data for which cumulants are calculated, n - order of highest comulant

output: cumulants up to n (dict of arrays)
"""
function snaivecumulant{T<:AbstractFloat}(data::Matrix{T}, n::Int)
    data = center(data)
    c2 = Base.covm(data, 0, 1, false)
    cumulants = [c2, moment_n(data, 3)]
    for k in 4:n
      c = moment_n(data, k) - produ(cumulants, k)
      push!(cumulants, c)
    end
    return cumulants
end
