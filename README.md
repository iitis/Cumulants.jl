# Cumulants.jl
[![Build Status](https://travis-ci.org/ZKSI/Cumulants.jl.svg?branch=master)](https://travis-ci.org/ZKSI/Cumulants.jl)
[![Coverage Status](https://coveralls.io/repos/github/ZKSI/Cumulants.jl/badge.svg?branch=master)](https://coveralls.io/github/ZKSI/Cumulants.jl?branch=master)

Calculates cummulant tensors of any order for multivariate data.
Functions return tensor or array of tensors in `SymmetricTensors` type. Requires [SymmetricTensors.jl](https://github.com/ZKSI/SymmetricTensors.jl). To convert to array, run:

```julia
julia> convert(Array, data::SymmetricTensors{T, N})
```

As of 01/01/2017 [kdomino](https://github.com/kdomino) is the lead maintainer of this package.

## Instalation

Within Julia, run

```julia
julia> Pkg.add("Cumulants")
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

[100.0 230.0; 230.0 560.0] NULL; NULL NULL]

Nullable{Array{Float64,3}}[[155.0 360.0; 360.0 890.0] [565.0; 1420.0]; NULL [2275.0]], 2, 2, 3, false)

```
To convert to array use `convert`

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
SymmetricTensors.SymmetricTensor{Float64,2}(Nullable{Array{Float64,2}}[[2.0 2.0; 2.0 2.0] [2.0; 2.0]; NULL [2.0]], 2, 2, 3, false)

julia> c[3]
SymmetricTensors.SymmetricTensor{Float64,3}(Nullable{Array{Float64,3}}[[0.0 0.0; 0.0 0.0]

[0.0 0.0; 0.0 0.0] NULL; NULL NULL]

Nullable{Array{Float64,3}}[[0.0 0.0; 0.0 0.0] [0.0; 0.0]; NULL [0.0]], 2, 2, 3, false)
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
julia> {T <: AbstractFloat}naivemoment(data::Matrix{T}, m::Int = 4)
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
julia> naivecumulant{T <: AbstractFloat}(data::Matrix{T}, m::Int = 4)
```
Returns `Array{T, m}` of the `m`'th cumulant of data, calculated using a naive algorithm. Works for `1 <= m , 7`, for `m >= 7` throws exception.


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

To analyse the computional time of cumulants vs naivecumulants and moment vs naivemoment, we supply the executable script `comptimes.jl`.
This script returns to a .jld file computional times, given folowing parameters:
* `-m (Int)`: cumulant's order, by default `m = 4`,
* `-n (vararg Int)`: numbers of marginal variables, by default `m = 20 24 28`,
* `-t (vararg Int)`: number of realistations of random variable, by defalut `t = 10000`.
Be carefull while using `n`>`4` and large `m`, where naive algorithms might need a large computional time and memory usage. Naive algorithms does not use the block structures, hence they computes and stores a whole cumulant tensor regardless its symmetry. All comparisons performed by this script use one core.

To analyse the computional time of cumulants for diferent block sizes `1 < b < sqrt(n)`, we supply the executable script `comptimes.jl`.
This script returns to a .jld file computional times, given folowing parameters:
* `-m (Int)`: cumulant's order, by default `m = 4`,
* `-n (Int)`: numbers of marginal variables, by default `m = 48`,
* `-t (vararg Int)`: number of realistations of random variable, by defalut `t = 10000 20000`.
Computional times and parameters are saved in the .jld file in /res directory. All comparisons performed by this script use one core.

To analyse the computional time of cumulants on different numbers of proseses, we supply the executable script `comptimeprocs.jl`.
This script returns to a .jld file computional times, given folowing parameters:
* `-m (Int)`: cumulant's order, by default `m = 4`,
* `-n (Int)`: numbers of marginal variables, by default `m = 50`,
* `-t (Int)`: number of realistations of random variable, by defalut `t = 100000`,
* `-p (Int)`: maximal number of proceses, by default `p = 4`,
* `-b (Int)`: blocks size, by default `b = 2`.

All result files are saved in /res directory. To plot a graph run /res/plotcomptimes.jl, with parameter
* `-f (String)`: file name without the .jld extension.

For the computional example on data use the following.

The script `gandata.jl` generates `t = 150000000` realisations of `n = 4` dimensional data form the `t`-multivariate distribution with `\nu = 14` degrees of freedom, and theoretical 
super-diagonal elements of those cumulants. Rasults are saved in `data/datafortests.jld`

The script `testondata.jl` computes cumulant tensors of order `m = 1 - 6` for `data/datafortests.jld`, results are saved in `data/cumulants.jld`.

To read `cumulants.jld` please run 

```julia 
julia> using JLD

julia> using SymmetricTensors

julia> load("cumulants.jld")

```

To plot super-diagonal elements of those cumulants and their theoretical values from t-student dostrobution pleas run `plotsuperdiag.jl`


# Citing this work


Krzysztof Domino, Piotr Gawron, Łukasz Pawela, *The tensor network representation of high order cumulant and algorithm for their calculation*, [arXiv:1701.05420](https://arxiv.org/abs/1701.05420)
