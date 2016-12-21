# ---- following code is used to caclulate moments in SymmetricTensor form ----
""" Center matrix columns (substracts column's mean).

Input: data in a matrix form.

Returns: matrix with centred columns.
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

"""Splits imput data into vector of arrays, each array is and provides data input
to corresponding block. Width of an array is s, widht of last the array
may be smaller if s / M

Input: data - matrix, s::Int - size of a block

Returns: vector of matrices
"""
function splitdata{T<:AbstractFloat}(data::Matrix{T}, s::Int)
  M = size(data,2)
  sizetest(M, s)
  ret = Matrix{T}[]
  for i in 1:ceil(Int, M/s)
    push!(ret, data[:,seg(i, s, M)])
  end
  ret
end

""" Calculates the single element of n'th moment tensor, from data in n long
vector of matrices at multi-index ind.

Input: Y - vector of matrices, ind - tuple (milti-index).

Returns: Float64 - n'th moment element.
"""
mom_el{T <: AbstractFloat}(Y::Vector{Matrix{T}}, ind::Tuple) =
      mean(mapreduce(i -> Y[i][:,ind[i]], .*, 1:length(ind)))

"""Calculate a single block of moment's tensor.

Input: Y - vector of matrices of data, n - int (moment's order).

Returns: a block, n dimentional Array{N}.
"""
function momentseg{T <: AbstractFloat}(Y::Vector{Matrix{T}}, n::Int)
  dims = map(j -> size(Y[j], 2), (1:n...))
  ret = (nprocs()== 1)? zeros(T, dims): SharedArray(T, dims)
  @sync @parallel for i = 1:prod(dims)
    ind = ind2sub(dims, i)
    @inbounds ret[ind...] = mom_el(Y, ind)
  end
  Array(ret)
end

"""Calculate n'th moment in SymmetricTensor form.

Input: X - vector of matrices of data, n - moment's order.

Returns: n - dimentional tensor in SymmetricTensor form.
"""
function momentbs{T <: AbstractFloat}(X::Vector{Matrix{T}}, n::Int)
    g = length(X)
    ret = NullableArray(Array{T, n}, fill(g, n)...)
    for i in indices(n, g)
      Y = map(k -> X[i[k]], 1:n)
      @inbounds ret[i...] = momentseg(Y, n)
    end
    SymmetricTensor(ret, false)
end

# ---- following code is used to caclulate cumulants in SymmetricTensor form----

"""Calculates outer product of segments, given partition of indices.

Input: s - Int, size of segment, part - (Vector{Vector}) - partition of indices,
ar_s - array of segments.

Returns: Array of size s^n.
"""
function prodblocks{T <: AbstractFloat}(s::Int, n::Int, part::Vector{Vector{Int}},
  ar_s::Vector{Array{T}})
  ret = (nprocs()== 1)? zeros(T, fill(s, n)...): SharedArray(T, fill(s, n)...)
  r = length(part)
  @sync @parallel for i = 1:(s^n)
    ind = ind2sub((fill(s, n)...), i)
    @inbounds ret[ind...] = mapreduce(i -> ar_s[i][ind[part[i]]...], *, 1:r)
  end
Array(ret)
end

"""Create all partitions of sequence 1:n into sigma subs multi-indices
with at least 2 elements each.

Returns: part_set - Vector{Vector{Vector}}, set of partitions
"""
function indpart(n::Int, sigma::Int)
    part_set = Vector{Vector{Int64}}[]
    for part in partitions(1:n, sigma)
      s = map(length, part)
      if !(1 in s)
        push!(part_set, part)
      end
    end
    part_set
end

"""Read blocks from diferent cumulants, given multiindex tuple i and its partition
part (Vector{Vector{Int}}). Size of particular inner vector gives order of
cumulant form wich block is read, values on inner vector gives partial permutation
of multiindex i.
Function val(bs, i) from symmetric tensors reads block from bs at multi-index i.

Further input: nsq - (Bool) determines if the block is squared. If not, slices
of zeros are added to make each its size equal to s.

Returns: vector blocks corrsponding to the partition. Size of a vector is such as
a size of outer vector of part, ndims of each block is such as corresponding size
of inner vector of part.
"""
function read{T <: AbstractFloat}(i::Tuple, part::Vector{Vector{Int}}, s::Int,
  nsq::Bool, st::SymmetricTensor{T}...)
  sigma = length(part)
  ret = Array(Array{T}, sigma)
  for k in 1:sigma
    N = length(part[k])
    data = val(st[N-1], i[part[k]])
    if nsq
      ind = map(k -> 1:size(data,k), 1:N)
      temp = zeros(T, fill(s, N)...)
      @inbounds temp[ind...] = data
      ret[k] = temp
    else
      ret[k] = data
    end
  end
  ret
end

"""Calculates the outer products of sigma cumulants for n order cumulant calculation.

Input: n Int, sigma Int,
c - (vararg) cumulants of order 2, ..., n-2 in SymmetricTensor form.

Output: n dims tensor in SymmetricTensor form.
"""
function outerp{T <: AbstractFloat}(n::Int, sigma::Int, c::SymmetricTensor{T}...)
  s,g,M = size(c[1])
  part = indpart(n, sigma)
  ret = NullableArray(Array{T, n}, fill(g, n)...)
  for i in indices(n, g)
    temp = zeros(T, fill(s, n)...)
    nsq = !c[1].sqr && (g in i)
    for p in part
      block_ar = read(i, p, s, nsq, c...)
      @inbounds temp += prodblocks(s, n, p, block_ar)
    end
    if nsq # cuts zeros
      range = map(k -> g==i[k]? (1:M%s): (1:s), 1:n)
      @inbounds temp = temp[range...]
    end
    @inbounds ret[i...] = temp
  end
  SymmetricTensor(ret, false)
end

"""Calculates n'th cumulant.

Input: X - vector of matrices of data, n int,
c - (vararg) cumulants of order 2, ..., n-2 in SymmetricTensor form

Output: n th cumulant in SymmetricTensor form (dims = n)."""
function cumulantn{T <: AbstractFloat}(X::Vector{Matrix{T}}, n::Int, c::SymmetricTensor{T}...)
  ret =  momentbs(X, n)
  for sigma in 2:floor(Int, n/2)
    ret -= outerp(n, sigma, c...)
  end
  ret
end

"""Recursive formula, calculate cumulants of order 2 - n.

Input: X - matrix of rough data, n int, s - block size.

Returns array of cumullants of order 2 - n, in SymmetricTensor form."""
function cumulants{T <: AbstractFloat}(X::Matrix{T}, n::Int, s::Int = 2)
  X = splitdata(center(X), s)
  ret = Array(SymmetricTensor{T}, n-1)
  for i = 2:n
    @inbounds ret[i-1] = (i < 4)? momentbs(X, i): cumulantn(X, i, ret[1:(i-3)]...)
  end
  ret
end
