# ---- following code is used to caclulate moments ----

""" center data. Given data matrix centers each column,
substracts columnwise mean for all data in the column
performs centring for each column,

Returns matrix
"""
function center!{T<:AbstractFloat}(data::Matrix{T})
  for i = 1:size(data, 2)
    @inbounds data[:,i] = data[:,i]-mean(data[:,i])
  end
end

function center{T<:AbstractFloat}(data::Matrix{T})
  centered = copy(data)
  center!(centered)
  centered
end

""" calculates the single element of the block of N'th moment

input vectors that corresponds to given column of data

Returns Float64 (an element of the block)
"""
momentel{T <: AbstractFloat}(v::Vector{T}...) = mean(mapreduce(i -> v[i], .*, 1:length(v)))

"""calculate n'th moment for the given segment

input r - matrices of data

Returns N dimentional array (segment)
"""
function momentseg{T <: AbstractFloat}(dims::Tuple, Y::Matrix{T}...)
  n = length(Y)
  ret = zeros(T, dims)
  for i = 1:prod(dims)
    ind = ind2sub((dims), i)
    @inbounds ret[ind...] = momentel(map(k -> Y[k][:,ind[k]], 1:n)...)
  end
  ret
end

""" calculate N'th moment in the bs form

input matrix of data, the order of the moment (N), number of segments for bs

Returns N dimentional Box structure of N'th moment

Case if last boxes are not squared
"""
function momentbs{T <: AbstractFloat}(X::Matrix{T}, n::Int, s::Int)
    M = size(X,2)
    g = ceil(Int, M/s)
    range = ((1:n)...)
    ret = NullableArray(Array{T, n}, fill(g, n)...)
    for i in indices(n, g)
      Y = map(k -> X[:,seg(i[k], s, M)], 1:n)
      dims = map(i -> (size(Y[i], 2)), range)
      @inbounds ret[i...] = momentseg(dims, Y...)
    end
    SymmetricTensor(ret)
end

"""split indices into given permutation of partitions

input: n , array of indices e.g. [i_1, i_2, i_3, i_4, i_5]
permutation of partitions represented by followign integers e.g. [[2,3],[1,4,5]]

Returns output e.g. [[i_2 i_3][i_1 i_4 i_5]]
"""
splitind(n::Vector{Int}, pe::Vector{Vector{Int}}) = map(p->n[p], pe)


"""calculates outer product of segments for given partition od indices

input s - size of segment, N - required number of dinsions of output, part - vector of partations (vectors of ints)
c - arrays of boxes

Returns N dimentional array of size s x .... x s
"""
function prodblocks{T <: AbstractFloat}(s::Int, n::Int, part::Vector{Vector{Int}}, c::Array{T}...)
  ret = zeros(T, fill(s, n)...)
  for i = 1:(s^n)
    ind = ind2sub((fill(s, n)...), i)
    pe = splitind(collect(ind), part)
    @inbounds ret[ind...] = mapreduce(i -> c[i][pe[i]...], *, 1:size(part, 1))
  end
  ret
end


""" given to multiindex lenght n and number of subsets omega
provides all partitions of the sequence 1:n into sigma subsets such that each
subset size is greater than 1 and subsets are disjoint
"""

function indpart(n::Int, sigma::Int)
    p = Vector{Vector{Int}}[]
    r = Vector{Int}[]
    for part in partitions(1:n, sigma)
      s = map(length, part)
      if !(1 in s)
        push!(p, part)
        push!(r, s)
      end
    end
    p, r, length(r)
end

"""if box is notsquared makes it square by adding slices with zeros

input the box array and reguired size

Returns N dimentional s x ...x s array
"""
function addzeros{T <: AbstractFloat, N}(s::Int, inputbox::Array{T,N})
  if !all(collect(size(inputbox)) .== s)
    ret = zeros(T, fill(s, N)...)
    ind = map(k -> 1:size(inputbox,k), 1:N)
    ret[ind...] = inputbox
    return ret
  end
  inputbox
end


"""
calculates mixed element for given sigma, if last blockes are not squared

"""
function outerp{T <: AbstractFloat}(n::Int, sigma::Int, c::SymmetricTensor{T}...)
  s,g,M = size(c[1])
  p, r, len = indpart(n, sigma)
  ret = NullableArray(Array{T, n}, fill(g, n)...)
  issquare = (s*g==M)
  for i in indices(n, g)
    temp = zeros(T, fill(s, n)...)
    for j in 1:len
      pe = splitind(collect(i), p[j])
      if !issquare && (g in i)
        @inbounds temp += prodblocks(s, n, p[j], map(l -> addzeros(s[1], c[r[j][l]-1].frame[pe[l]...].value), 1:sigma)...)
      else
        @inbounds temp += prodblocks(s, n, p[j], map(l -> c[r[j][l]-1].frame[pe[l]...].value, 1:sigma)...)
      end
    end
    if !issquare && (g in i)
      range = map(k -> ((g == i[k])? (1:(M%s)) : (1:s)), (1:length(i)))
      temp = temp[range...]
    end
    @inbounds ret[i...] = temp
  end
  SymmetricTensor(ret)
end


"""calculates n'th cumulant,

input data - matrix of data, n - the order of the cumulant, segments - number of segments for bs
c - cumulants in the bs form orderred as follow c2, c3, ..., c(n-2)

Returns the n order cumulant in the bs form"""
function cumulantn{T <: AbstractFloat}(X::Matrix{T}, n::Int, s::Int, c::SymmetricTensor{T}...)
  ret =  momentbs(X, n, s)
  for sigma in 2:floor(Int, n/2)
    ret -= outerp(n, sigma, c...)
  end
  ret
end


"""recursive formula, calculate cumulants up to order n

input: data - matrix of data, n - the maximal order of the cumulant, segments - number of segments for bs

Returns cumulants in the bs form orderred as follow c2, c3, ..., cn
works for any n >= 2, tested up to n = 10, in automatic tests up to n = 6 (limit due to the increasement
in computation time for benchmark algorithm (semi naive))
"""
function cumulants{T <: AbstractFloat}(X::Matrix{T}, n::Int, s::Int = 3)
  X = center(X)
  ret = Array(SymmetricTensor{T}, n-1)
  for i = 2:n
    ret[i-1] =  (i < 4)? momentbs(X, i, s) : cumulantn(X, i, s, ret[1:(i-3)]...)
  end
  ret
end
