## following code is used to caclulate moments in SymmetricTensor form ##

"""

    blockel(X::Matrix{T}, i::Tuple, j::Tuple, b::Int)

Returns Float, the element of the block (indexed by j) of the moment's tensor
 of X, at index cd i inside a block, where b is a standard blocks' size


```jldoctest
julia> M = [1. 2.  5. 6. ; 3. 4.  7. 8.];

julia> mom_el(M, (1,1), (1,1), 2)
5.0

julia> mom_el(M, (1,1), (2,2), 2)
37.0
 ```
"""

blockel{T <: AbstractFloat}(X::Matrix{T}, i::Tuple, j::Tuple, b::Int) =
    mean(mapreduce(k::Int -> X[:,(j[k]-1)*b+i[k]], .*, 1:length(i)))


"""

  momentblock(X::Matrix{T}, j::Tuple, dims::Tuple, b::Int)

Returns a block of a moment's tensor of X. A block is indexed by j and if size dims,
b is a standatd block size.

```jldoctest
julia> M = [1. 2.  5. 6. ; 3. 4.  7. 8.];

julia> momentblock(M, (1,1), (2,2), 2)
2×2 Array{Float64,2}:
 5.0   7.0
 7.0  10.0
```
"""

function momentblock{T <: AbstractFloat}(X::Matrix{T}, j::Tuple, dims::Tuple, b::Int)
  ret = zeros(T, dims)
  for ind = 1:(prod(dims))
    i = ind2sub(dims, ind)
    @inbounds ret[i...] = blockel(X, i, j, b)
  end
  Array(ret)
end

"""
    moment(X::Matrix}, m::Int, b::Int)

Returns: SymmetricTensor{Float, m}, a tensor of the m'th moment of X, where b
is a block size.
"""
function moment{T <: AbstractFloat}(X::Matrix{T}, m::Int, b::Int=2)
  n = size(X, 2)
  sizetest(n, b)
  nbar = ceil(Int, n/b)
  ret = NullableArray(Array{T, m}, fill(nbar, m)...)
  for j in indices(m, nbar)
    dims = (mod(n,b) == 0 || !(nbar in j))? (fill(b,m)...): usebl(j, n, b, nbar)
    @inbounds ret[j...] = momentblock(X, j, dims, b)
  end
  SymmetricTensor(ret; testdatstruct = false)
end


"""
    usebl(bind::Tuple, n::Int, b::Int, nbar::Int)

Returns: Tuple{Int}, sizes of the last block
"""
function usebl(bind::Tuple, n::Int, b::Int, nbar::Int)
  bl = n - b*(nbar-1)
  map(i -> (i == nbar)? (bl) : (b), bind)
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
  b = size(blocks[1], 1)
  block = zeros(T, fill(b, inp.nind)...)
  for i = 1:(b^inp.nind)
    muli = ind2sub((fill(b, inp.nind)...), i)
    @inbounds block[muli...] =
    mapreduce(i -> blocks[i][muli[inp.part[i]]...], *, 1:inp.npart)
  end
  block
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

    cumulant(X::Vector{Matrix}, cum::SymmetricTensor...)

Returns: SymmetricTensor{Float, m}, a tensor of the m'th cumulant of X, given Vector
of cumulants of order 2, ..., m-2
"""

function cumulant{T <: AbstractFloat}(X::Matrix{T}, cum::SymmetricTensor{T}...)
  m = length(cum) + 3
  ret =  moment(X, m, cum[1].bls)
  for sigma in 2:floor(Int, m/2)
    ret -= outerpodcum(m, sigma, cum...)
  end
  ret
end

"""

    cumulants(X::Matrix, m::Int, b::Int)

Returns [SymmetricTensor{Float, 2}, ..., SymmetricTensor{Float, m}], vector of
cumulants tensors

```
julia> M =  [[-0.88626   0.279571];[-0.704774  0.131896]];

julia> convert(Array, cumulants(M, 3)[2])
2×2×2 Array{Float64,3}:
[:, :, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2] =
 0.0  0.0
 0.0  0.0
```
"""
function cumulants{T <: AbstractFloat}(X::Matrix{T}, m::Int = 4, b::Int = 2)
  X = X .- mean(X, 1)
  cvec = Array(SymmetricTensor{T}, m-1)
  for i = 2:m
    @inbounds cvec[i-1] = (i < 4)? moment(X, i, b): cumulant(X, cvec[1:(i-3)]...)
  end
  cvec
end

"""

    getcumulant(cum::Vector{SymmetricTensor{T}}, m::Int)

Returns SymmetricTensor{Float, m}, the m'th cumulant tensor
"""
function getcumulant{T <: AbstractFloat}(cum::Vector{SymmetricTensor{T}}, m::Int)
  if m == 1
    throw(BoundsError("mean vector not stored"))
  else
    return cum[m-1]
  end
end
