addprocs(3)
procs()


@everywhere using SymmetricTensors
@everywhere using NullableArrays
@everywhere import SymmetricTensors: ind2range, indices
@everywhere using Cumulants
#using Cumulants

ind2range(2,1000,2000)


a = randn(10000, 10000);

@everywhere function ff(x::Float64)
  for i in 1:1000000000
    log(x)
  end
  1.
end


x = randn(120000000, 2)



@everywhere function f(x::Matrix{Float64})
  y = x[:,1]
  z = x[:,2]
  [mean(y.^6),  mean(z.^6)]
end


@everywhere blockel{T <: AbstractFloat}(X::Matrix{T}, i::Tuple, j::Tuple, b::Int) =
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

@everywhere function momentblock{T <: AbstractFloat}(X::Matrix{T}, j::Tuple, dims::Tuple, b::Int)
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
@everywhere function moment{T <: AbstractFloat}(X::Matrix{T}, m::Int, b::Int=2)
  n = size(X, 2)
  #sizetest(n, b)
  nbar = ceil(Int, n/b)
  ret = NullableArray(Array{T, m}, fill(nbar, m)...)
  for j in indices(m, nbar)
    dims = (mod(n,b) == 0 || !(nbar in j))? (fill(b,m)...): usebl(j, n, b, nbar)
    @inbounds ret[j...] = momentblock(X, j, dims, b)
  end
  SymmetricTensor(ret; testdatstruct = false)
end

function usebl(bind::Tuple, n::Int, b::Int, nbar::Int)
  bl = n - b*(nbar-1)
  map(i -> (i == nbar)? (bl) : (b), bind)
end

function ff(x::Matrix{Float64})
  T = size(x, 1)
  k = nprocs()
  r = ceil(Int, T/k)
  y = [x[ind2range(i, r, T), :] for i in 1:k]
  ret = pmap(f, y)
  r = ret.*[fill(r, k-1)..., (T-(k-1)*r)]
  [sum(r[1])/T, sum(r[2])/T]
end

ff(x) - f(x)

@time f(x)
@time ff(x)


a = randn(100000,42)

@time m = moment(a, 4)

nprocs()
rmprocs(2,3,4)

function fm(x::Matrix{Float64})
  @everywhere f44(x) = moment(x, 4)
  T = size(x, 1)
  k = nprocs()
  r = ceil(Int, T/k)
  y = [x[ind2range(i, r, T), :] for i in 1:k]
  ret = pmap(f44, y)
  (r*sum(ret[1:(end-1)])+(T-(k-1)*r)ret[end])/T
end

@time mm = fm(a)

maximum(abs(convert(Array, mm)- convert(Array, m)))



m*1.
