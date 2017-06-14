# Cumulants.jl
[![Build Status](https://travis-ci.org/ZKSI/Cumulants.jl.svg?branch=master)](https://travis-ci.org/ZKSI/Cumulants.jl)

Calculates Cummulant tensors of any order for multivariate data.
Functions return tensor or array of tensors in `SymmetricTensors` type. Requires [SymmetricTensors.jl](https://github.com/ZKSI/SymmetricTensors.jl). To convert to array, run:

```julia
julia> convert(Array, data::SymmetricTensors{T, N})
```

As of 01/01/2017 [kdomino](https://github.com/kdomino) is the lead maintainer of this package.

## Instalation

Within Julia, run

```julia
julia> Pkg.clone("Cumulants")
```

to install the files.  Julia 0.5 is required.


## Functions
### Moment

```julia
julia> moment{T <: AbstractFloat}(data::Matrix{T}, m::Int, b::Int = 2)
```

Returns a `SymmetricTensor{T, m}` of the moment of order `m` of multivariate data represented by a `t` by `n` matrix, i.e. data with `n` marginal variables and `t` realisations. The argument `b` with defalt value `2`, is an optional `Int` that determines the size
of the blocks in `SymmetricTensors` type.

```julia
julia> data = reshape(collect(1.:15.),(5,3))
5×3 Array{Float64,2}:
 1.0   6.0  11.0
 2.0   7.0  12.0
 3.0   8.0  13.0
 4.0   9.0  14.0
 5.0  10.0  15.0
```

```julia
julia> m = moment(data, 3)
SymmetricTensors.SymmetricTensor{Float64,3}(Nullable{Array{Float64,3}}[[45.0 100.0; 100.0 230.0]

[100.0 230.0; 230.0 560.0] #NULL; #NULL #NULL]

Nullable{Array{Float64,3}}[[155.0 360.0; 360.0 890.0] [565.0; 1420.0]; #NULL [2275.0]],2,2,3,false)

```
To convert to array just run convert

```julia
julia> convert(Array, m)
3×3×3 Array{Float64,3}:
[:, :, 1] =
  45.0  100.0  155.0
 100.0  230.0  360.0
 155.0  360.0  565.0

[:, :, 2] =
 100.0  230.0   360.0                                                                                                                                                       
 230.0  560.0   890.0                                                                                                                                                       
 360.0  890.0  1420.0                                                                                                                                                       

[:, :, 3] =                                                                                                                                                                 
 155.0   360.0   565.0                                                                                                                                                      
 360.0   890.0  1420.0                                                                                                                                               
 565.0  1420.0  2275.0
 ```

 ### Cumulants

 ```julia
julia> cumulants{T <: AbstractFloat}(data::Matrix{T}, m::Int = 4, b::Int = 2)
```

Returns a vector of `SymmetricTensor{T, i}` `i = 1,2,3,...,m` of cumulants of
order `1,2,3,...,m`. Cumulants are calculated for multivariate data represented
by matrix of size `t` by `n`, i.e. data with `n` marginal variables and `t`
realisations. The argument `b` with default value `2`, is an optional `Int`
that determines a size of blocks in `SymmetricTensors` type.

```julia
julia> c = cumulants(data, 3);

julia> c[2]
SymmetricTensors.SymmetricTensor{Float64,2}(Nullable{Array{Float64,2}}[[2.0 2.0; 2.0 2.0] [2.0; 2.0]; #NULL [2.0]],2,2,3,false)

julia> c[3]
SymmetricTensors.SymmetricTensor{Float64,3}(Nullable{Array{Float64,3}}[[0.0 0.0; 0.0 0.0]

[0.0 0.0; 0.0 0.0] #NULL; #NULL #NULL]

Nullable{Array{Float64,3}}[[0.0 0.0; 0.0 0.0] [0.0; 0.0]; #NULL [0.0]],2,2,3,false)
```
To convert to array given element of the vector `c`, just run:

```julia
julia> convert(Array, c[2])
3×3 Array{Float64,2}:
 2.0  2.0  2.0
 2.0  2.0  2.0
 2.0  2.0  2.0

 julia> convert(Array, c[3])
3×3×3 Array{Float64,3}:
[:, :, 1] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 3] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
```

Parallel computation available, it is efficient for large number of data realisations, e.g. `t = 1000000`. For parallel computation just run
```julia
julia> addprocs(n)
julia> @everywhere using Cumulants
```

Naive algorithms of moment and cumulant tesors calculations are also available.

 ```julia
julia> {T <: AbstractFloat}naivemoment(data::Matrix{T}, m::Int)
```
Returns array{T, m} of the m'th moment of data. calculated using a naive algorithm.


```julia
julia> naivemoment(data, 3)
3×3×3 Array{Float64,3}:
[:, :, 1] =
  45.0  100.0  155.0
 100.0  230.0  360.0
 155.0  360.0  565.0

[:, :, 2] =
 100.0  230.0   360.0
 230.0  560.0   890.0
 360.0  890.0  1420.0

[:, :, 3] =
 155.0   360.0   565.0
 360.0   890.0  1420.0
 565.0  1420.0  2275.0
```

 ```julia
julia> naivecumulant{T <: AbstractFloat}(data::Matrix{T}, m::Int)
```
Returns `Array{T, m}` of the `m`'th cumulant of data, calculated using a naive algorithm.


```julia
julia> naivecumulant(data, 2)
3×3 Array{Float64,2}:
 2.0  2.0  2.0
 2.0  2.0  2.0
 2.0  2.0  2.0
```


```julia
julia> naivecumulant(data, 3)
3×3×3 Array{Float64,3}:
[:, :, 1] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 2] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0

[:, :, 3] =
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
```

# Performance analysis

To analyse the computional time of our algorithm vs a naive algorithm and the algorithm introduced in 'MULTIVARIATE CUMULANTS IN R, 2012, JAN DE LEEUW' and implemented in Julia, we supply the executable script `comptimes.jl`.
This script returns charts of the computional speedup of our algorithm vs other algorithms described above.
It has optional arguments:
* `-m (Int)`: cumulant's order,
* `-n (vararg Int)`: numbers of marginal variables,
* `-t (vararg Int)`: number of realistations of random variable.
The default values are `t = 4000`, `n = 22 24` and `m = 4`. For good performance results use higher `t` and `n`, but the computional time of naive algorithm will be large. Be cautious while  using `m`>`4`, as the naive algorithm calculations may need a large time and there may be a memory shortage since the comparison algorithms does not use a block structure and stores the whole array in memory. All comparisons performed by this script use only one core.

The script `gandata.jl` generates `t = 75000000` realisations of `n = 4` variate data form the `t`-multivariate distribution with `\nu = 14` degrees of freedom. The script `testondata.jl` computes cumulant tensors of order `m = 2,3,...,6` of those data and displays some of cumulants valuse on charts. For superdiagonal values the comparison with theoretical cumulants values of the distrubution is supplied.

# Citing this work


Krzysztof Domino, Piotr Gawron, Łukasz Pawela, *The tensor network representation of high order cumulant and algorithm for their calculation*, [arXiv:1701.05420](https://arxiv.org/abs/1701.05420)
