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

Type that stores a partition of multiindex into subests, sizes of subests,
size of original multitindex and number of subsets

"""
type IndexPart
  part::Vector{Vector{Int64}}
  subsetslen::Vector{Int64}
  nind::Int
  npart::Int
  (::Type{IndexPart})(part::Vector{Vector{Int64}}, subsetslen::Vector{Int64},
nind::Int, npart::Int) = new(part, subsetslen, nind, npart)
end

"""

    indpart(nind::Int, npart::Int)

Returns vector of IndexPart type, that includes partitions of set [1, 2, ..., nind]
into npart subests of size >= 2, sizes of each subest, size of original set and
number of partitions

```jldoctest
julia>indpart(4,2)
3-element Array{Cumulants.IndexPart,1}:
 IndexPart(Array{Int64,1}[[1,2],[3,4]],[2,2],4,2)
 IndexPart(Array{Int64,1}[[1,3],[2,4]],[2,2],4,2)
 IndexPart(Array{Int64,1}[[1,4],[2,3]],[2,2],4,2)
```
"""

function indpart(nind::Int, npart::Int)
    part_set = IndexPart[]
    for part in partitions(1:nind, npart)
      subsetslen = map(length, part)
      if !(1 in subsetslen)
        push!(part_set, IndexPart(part, subsetslen, nind, npart))
      end
    end
    part_set
end

"""

  accesscum(mulind::Tuple{Int, ...}, ::IndexPart,
    cum::SymmetricTensor{Float}...)


Returns: vector of blocks from cumulants. Each block correspond to a subests
 of partition (part) of multiindex (multiind).

 ```jldoctest
 julia> cum = SymmetricTensor([1.0 2.0 3.0; 2.0 4.0 6.0; 3.0 6.0 5.0]);

julia> accesscum((1,1,1,1), IndexPart(Array{Int64,1}[[1,2],[3,4]],[2,2],4,2), cum)
Array{Float64,N}[
[1.0 2.0; 2.0 4.0],
[1.0 2.0; 2.0 4.0]]

julia> accesscum((1,1,1,2), IndexPart(Array{Int64,1}[[1,2],[3,4]],[2,2],4,2), cum)

Array{Float64,N}[
[1.0 2.0; 2.0 4.0],
[3.0 0.0; 6.0 0.0]]

julia> accesscum((1,1,1,1), IndexPart(Array{Int64,1}[[1,4],[2,3]],[2,2],4,2), cum)

Array{Float64,N}[
[1.0 2.0; 2.0 4.0],
[1.0 2.0; 2.0 4.0]]
 ```
"""
function accesscum{T <: AbstractFloat}(mulind::Tuple, part::IndexPart, cum::SymmetricTensor{T}...)
  blocks = Array(Array{T}, part.npart)
  for k in 1:part.npart
    data = cum[part.subsetslen[k]-1][mulind[part.part[k]]]
    if cum[1].sqr || !(cum[1].bln in mulind)
      blocks[k] = data
    else
      ind = map(i -> 1:size(data,i), 1:part.subsetslen[k])
      datapadded = zeros(T, fill(cum[1].bls, part.subsetslen[k])...)
      @inbounds datapadded[ind...] = data
      blocks[k] = datapadded
    end
  end
  blocks
end

"""

    outprodblocks(n::Int, part::Vector{Vector{Int}}, blocks::Vector{Array{T}}

Returns: n dims Array of outer product of blocks, given partition of indices, part.

```jldoctest
julia> blocks = 2-element Array{Array{Float64,N},1}[[1.0 2.0; 2.0 4.0], [1.0 2.0; 2.0 4.0]];

julia> outprodblocks(IndexPart(Array{Int64,1}[[1,2],[3,4]],[2,2],4,2), blocks)

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
function outprodblocks{T <: AbstractFloat}(inp::IndexPart, blocks::Vector{Array{T}})
  s = size(blocks[1], 1)
  block = (nprocs()== 1)? zeros(T, fill(s, inp.nind)...): SharedArray(T, fill(s, inp.nind)...)
  @sync @parallel for i = 1:(s^inp.nind)
    muli = ind2sub((fill(s, inp.nind)...), i)
    @inbounds block[muli...] =
    mapreduce(i -> blocks[i][muli[inp.part[i]]...], *, 1:inp.npart)
  end
Array(block)
end

"""
    outerpodcum(retd::Int, npart::Int, cum::SymmetricTensor...)

Returns retd dims outer products of npart cumulants in SymmetricTensor form.

```jldoctest
julia> cum = SymmetricTensor([1.0 2.0 3.0; 2.0 4.0 6.0; 3.0 6.0 5.0]);

julia> outerpodcum(4,2,cum)
SymmetricTensor(Nullable{Array{Float64,4}}[[3.0 6.0; 6.0 12.0]

[6.0 12.0; 12.0 24.0]

[6.0 12.0; 12.0 24.0]

[12.0 24.0; 24.0 48.0] #NULL; #NULL #NULL]

Nullable{Array{Float64,4}}[#NULL #NULL; #NULL #NULL]

Nullable{Array{Float64,4}}[[9.0 18.0; 18.0 36.0]

[18.0 36.0; 36.0 72.0] #NULL; #NULL #NULL]

Nullable{Array{Float64,4}}[[23.0 46.0; 46.0 92.0] [45.0; 90.0]; #NULL [75.0]],2,2,3,false)
```
"""
function outerpodcum{T <: AbstractFloat}(retd::Int, npart::Int, cum::SymmetricTensor{T}...)
  parts = indpart(retd, npart)
  prodcum = NullableArray(Array{T, retd}, fill(cum[1].bln, retd)...)
  for muli in indices(retd, cum[1].bln)
    block = zeros(T, fill(cum[1].bls, retd)...)
    for part in parts
      blocks = accesscum(muli, part, cum...)
      @inbounds block += outprodblocks(part, blocks)
    end
    if !cum[1].sqr && cum[1].bln in muli
      ran = map(k->cum[1].bln == muli[k]? (1:cum[1].dats% cum[1].bls): (1:cum[1].bls), 1:retd)
      @inbounds block = block[ran...]
    end
    @inbounds prodcum[muli...] = block
  end
  SymmetricTensor(prodcum; testdatstruct = false)
end

"""

    cumulantn(X::Vector{Matrix}, n::Int, cum::SymmetricTensor...)

Returns n th cumulant given multivariate data stored in X and lower order cumulants
cum
"""

function cumulantn{T <: AbstractFloat}(X::Vector{Matrix{T}}, n::Int, cum::SymmetricTensor{T}...)
  ret =  momentbs(X, n)
  for sigma in 2:floor(Int, n/2)
    ret -= outerpodcum(n, sigma, cum...)
  end
  ret
end

"""

    cumulants(X::Matrix, n::Int, bls::Int)

Returns array of cumullants of order 2 - n, in the SymmetricTensor form.
"""
function cumulants{T <: AbstractFloat}(X::Matrix{T}, n::Int, bls::Int = 2)
  X = splitdata(center(X), bls)
  ret = Array(SymmetricTensor{T}, n-1)
  for i = 2:n
    @inbounds ret[i-1] = (i < 4)? momentbs(X, i): cumulantn(X, i, ret[1:(i-3)]...)
  end
  ret
end
