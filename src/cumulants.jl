## following code is used to caclulate moments in SymmetricTensor form ##
"""
  center(M::Matrix)

Returns a matrix with centred columns.

```jldoctest
julia> center([[1.   2.]; [2.  4.]])
2×2 Matrix:
 -0.5  -1.0
  0.5   1.0

```
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

"""

    splitdata(M::Matrix, s::Int)

Returns vector of matrices, each of bls columns. Mumber of columns
is smaller if ! bls / size(M, 2)

```jldoctest
julia> M = [1. 2. 3. 4.; 5. 6. 7. 8.];

julia> splitdata(M, 2)
2-element Array{Matrix,1}:
 [1.0 2.0; 5.0 6.0]
 [3.0 4.0; 7.0 8.0]

julia> splitdata(M, 3)
2-element Array{Matrix,1}:
 [1.0 2.0 3.0; 5.0 6.0 7.0]
 [4.0; 8.0]
```
"""
function splitdata{T<:AbstractFloat}(data::Matrix{T}, bls::Int)
  M = size(data,2)
  sizetest(M, bls)
  ret = Matrix{T}[]
  for i in 1:ceil(Int, M/bls)
    push!(ret, data[:,ind2range(i, bls, M)])
  end
  ret
end

"""

    mom_el(Y::Vector{Matrix}, multind::Tuple{Int...})

Returns AbstractFloat, the mean of elementwise multiple of particular columns
(one column from each matrix in Y). For k'th matrix (Y[k]) the number of the
column is determined by k'th element of multind.

```jldoctest
julia> Y = [[1. 2. ; 5. 6.],[3. 4. ; 7. 8.]];

julia> mom_el(Y, (1,1))
19.0

julia> mom_el(Y, (1,2))
22.0
 ```
"""
mom_el{T <: AbstractFloat}(Y::Vector{Matrix{T}}, mulind::Tuple) =
      mean(mapreduce(i -> Y[i][:,mulind[i]], .*, 1:length(mulind)))

"""

  momentseg(Y::Vector{Matrix})

Returns a block - Array{AbstractFloat, length(Y)} that is a reduction of matrices in Y

```jldoctest
julia> momentseg([[1. 2. ; 5. 6.],[3. 4. ; 7. 8.]])
2×2 Matrix:
 19.0  22.0
 24.0  28.0
```
"""
function momentseg{T <: AbstractFloat}(Y::Vector{Matrix{T}})
  dims = map(j -> size(Y[j], 2), (1:length(Y)...))
  ret = (nprocs()== 1)? zeros(T, dims): SharedArray(T, dims)
  @sync @parallel for i = 1:prod(dims)
    mulind = ind2sub(dims, i)
    @inbounds ret[mulind...] = mom_el(Y, mulind)
  end
  Array(ret)
end

"""
    momentbs(X::Vector{Matrix, n::Int)

Returns: outdims - dimentional moment tensor in SymmetricTensor form.
```jldoctest
julia> mom = momentbs([[1. 2. ; 5. 6.],[3. 4. ; 7. 8.]], 2);

julia> mom.frame
2×2 NullableArray{Matrix, 2}:
 [13.0 16.0; 16.0 20.0]  [19.0 22.0; 24.0 28.0]
 #NULL                   [29.0 34.0; 34.0 40.0]

```

"""
function momentbs{T <: AbstractFloat}(X::Vector{Matrix{T}}, outdims::Int)
    ret = NullableArray(Array{T, outdims}, fill(length(X), outdims)...)
    for mulind in indices(outdims, length(X))
      Y = map(k -> X[mulind[k]], 1:outdims)
      @inbounds ret[mulind...] = momentseg(Y)
    end
    SymmetricTensor(ret; testdatstruct = false)
end

# ---- following code is used to caclulate cumulants in SymmetricTensor form----

"""

    indpart(nind::Int, npart::Int)

Returns vector all partitions of set [1, 2, ..., nind] into npart of size >= 2
each.

```jldoctest
julia>indpart(4,2)
3-element Array{Array{Array{Int,1},1},1}:
[[1,2],[3,4]]
[[1,3],[2,4]]
[[1,4],[2,3]]
```
"""
function indpart(nind::Int, npart::Int)
    part_set = Vector{Vector{Int64}}[]
    for part in partitions(1:nind, npart)
      s = map(length, part)
      if !(1 in s)
        push!(part_set, part)
      end
    end
    part_set
end

"""

  accesscum(mulind::Tuple{Int, ...}, part::Vector{Vector{Int}}, sqrblk::Bool,
    cum::SymmetricTensor{Float}...)


Returns: vector of blocks (arrays) corrsponding to the partition (part) of multiindex.
Each part of the partition indexes a block from cumulant of order equal to lenght(part).
 sqrblk - (Bool) determines if the block is squared. If not, slices of zeros are
 added to make each its size equal to s.

 ```jldoctest
 julia> cum = SymmetricTensor([1.0 2.0 3.0; 2.0 4.0 6.0; 3.0 6.0 5.0]);

julia> accesscum((1,1,1,1), [[1,2],[3,4]], false, cum)
Array{Float64,N}[
[1.0 2.0; 2.0 4.0],
[1.0 2.0; 2.0 4.0]]

julia> accesscum((1,1,1,2), [[1,2],[3,4]], false, cum)

Array{Float64,N}[
[1.0 2.0; 2.0 4.0],
[3.0 0.0; 6.0 0.0]]

julia> accesscum((1,1,1,1), [[1,4],[2,3]], false, cum)

Array{Float64,N}[
[1.0 2.0; 2.0 4.0],
[1.0 2.0; 2.0 4.0]]
 ```
"""
function accesscum{T <: AbstractFloat}(mulind::Tuple, part::Vector{Vector{Int}},
  sqrblk::Bool, cum::SymmetricTensor{T}...)
  npart = length(part)
  blocks = Array(Array{T}, npart)
  for k in 1:npart
    N = length(part[k])
    data = accessblock(cum[N-1], mulind[part[k]])
    if sqrblk
      blocks[k] = data
    else
      ind = map(k -> 1:size(data,k), 1:N)
      datapadded = zeros(T, fill(cum[1].bls, N)...)
      @inbounds datapadded[ind...] = data
      blocks[k] = datapadded
    end
  end
  blocks
end

"""Calculates outer product of blocks, given partition of indices.

Input: s - Int, size of segment, part - (Vector{Vector}) - partition of indices,
ar_s - array of segments.

Returns: Array of size s^n.

```jldoctest
julia> blocks = 2-element Array{Array{Float64,N},1}[[1.0 2.0; 2.0 4.0], [1.0 2.0; 2.0 4.0]];

julia> outprodblocks(4, [[1,2],[3,4]], a)

2×2×2×2 Array{Float64,4}:
[:, :, 1, 1] =
 1.0  2.0
 2.0  4.0

[:, :, 2, 1] =
 2.0  4.0
 4.0  8.0

[:, :, 1, 2] =
 2.0  4.0
 4.0  8.0

[:, :, 2, 2] =
 4.0   8.0
 8.0  16.0
```
"""
function outprodblocks{T <: AbstractFloat}(n::Int, part::Vector{Vector{Int}},
  blocks::Vector{Array{T}})
  s = size(blocks[1], 1)
  npart = length(part)
  block = (nprocs()== 1)? zeros(T, fill(s, n)...): SharedArray(T, fill(s, n)...)
  @sync @parallel for i = 1:(s^n)
    mulind = ind2sub((fill(s, n)...), i)
    @inbounds block[mulind...] = mapreduce(i -> blocks[i][mulind[part[i]]...], *, 1:npart)
  end
Array(block)
end

"""Calculates the outer products of sigma number of cumulants for n order cumulant
  calculation.

Input: n Int, sigma Int,
c - (vararg) cumulants of order 2, ..., n-2 in SymmetricTensor form.

Output: n dims tensor in SymmetricTensor form.
"""
function outerpodcum{T <: AbstractFloat}(outdims::Int, npart::Int, cum::SymmetricTensor{T}...)
  parts = indpart(outdims, npart)
  prodcum = NullableArray(Array{T, outdims}, fill(cum[1].bln, outdims)...)
  for mulind in indices(outdims, cum[1].bln)
    block = zeros(T, fill(cum[1].bls, outdims)...)
    sqrblk = cum[1].sqr || !(cum[1].bln in mulind)
    for part in parts
      blocks = accesscum(mulind, part, sqrblk, cum...)
      @inbounds block += outprodblocks(outdims, part, blocks)
    end
    if !sqrblk # cuts zeros
      range = map(k->cum[1].bln == mulind[k]? (1:cum[1].dats% cum[1].bls): (1:cum[1].bls), 1:outdims)
      @inbounds block = block[range...]
    end
    @inbounds prodcum[mulind...] = block
  end
  SymmetricTensor(prodcum; testdatstruct = false)
end

"""Calculates n'th cumulant.

Input: X - vector of matrices of data, n int,
c - (vararg) cumulants of order 2, ..., n-2 in SymmetricTensor form

Output: n th cumulant in SymmetricTensor form (dims = n)."""
function cumulantn{T <: AbstractFloat}(X::Vector{Matrix{T}}, n::Int, c::SymmetricTensor{T}...)
  ret =  momentbs(X, n)
  for sigma in 2:floor(Int, n/2)
    ret -= outerpodcum(n, sigma, c...)
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
