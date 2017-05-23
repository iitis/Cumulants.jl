# Cumulants.jl

Calculates Cummulant tensors of any order for multivariate data. 
Functions always return tensor or array of tensors in `SymmetricTensors` type. Raquires SymmetricTensors.jl "https://github.com/kdomino/SymmetricTensors.jl". To convert to array, run:

```julia
julia> convert(Array, data::SymmetricTensors{T, N})
```

As of 01/01/2017 "https://github.com/kdomino" is the lead maintainer of this package.

## Instalation

Within Julia, just use run 

```julia
julia> Pkg.clone("https://github.com/kdomino/Cumulants.jl")
```

to install the files.  Julia 0.5 or later required.


## Functions
### Moment

```julia
julia> moment{T <: AbstractFloat}(data::Matrix{T}, m::Int, b::Int = 2)
```

Returns a tensor of the moment of the order m of multivariate data.
For multivariate data takes matrix, where columns numerates marginal variables and rows 
numertates their realisations, b is an optional Int that determines a size 
of blocks in `SymmetricTensors` type.

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

Returns a vector of tensors of cumulants of order 2,3,...,m of multivariate data.
Input the same as for the moment function. Returns a vector of `SymmetricTensors`, with blocks of size b.

```julia
julia> c = cumulants(data, 3);

julia> c[1]
SymmetricTensors.SymmetricTensor{Float64,2}(Nullable{Array{Float64,2}}[[2.0 2.0; 2.0 2.0] [2.0; 2.0]; #NULL [2.0]],2,2,3,false)

julia> c[2]
SymmetricTensors.SymmetricTensor{Float64,3}(Nullable{Array{Float64,3}}[[0.0 0.0; 0.0 0.0]

[0.0 0.0; 0.0 0.0] #NULL; #NULL #NULL]

Nullable{Array{Float64,3}}[[0.0 0.0; 0.0 0.0] [0.0; 0.0]; #NULL [0.0]],2,2,3,false)
```
To convert to array given element ot the vector c, just run:

```julia
julia> convert(Array, c[1])
3×3 Array{Float64,2}:
 2.0  2.0  2.0
 2.0  2.0  2.0
 2.0  2.0  2.0

 julia> convert(Array, c[2])
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

To access cumulant of given order, use

```julia
julia> getcumulant{T <: AbstractFloat}(c::Vector{SymmetricTensor{T}}, order::Int)
```

```julia
julia> convert(Array, getcumulant(c, 2))
3×3 Array{Float64,2}:
 2.0  2.0  2.0
 2.0  2.0  2.0
 2.0  2.0  2.0
 
 julia> convert(Array, getcumulant(c, 3))
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
 
 julia> convert(Array, getcumulant(c, 1))
ERROR: BoundsError: attempt to access "mean vector not stored"
```
For parallel computation just run 
```julia
julia> addprocs()
julia> using Cumulants
julia> rmprocs()
```

The naive algorithms of moment and cumulant tesors calculations are also available. 

 ```julia
julia> {T <: AbstractFloat}naivemoment(data::Matrix{T}, m::Int)
```
Returns array{T, m} of the m'th moment calculated using a naive algorithm. 


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
julia> {T <: AbstractFloat}naivecumulant(data::Matrix{T}, m::Int)
```
Returns array{T, m} of the m'th cumulant calculated using a naive algorithm. 


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


To analyse the computional time of introduced there algorithm vs the naive algorithm and the algorithm
from 'MULTIVARIATE CUMULANTS IN R, 2012, JAN DE LEEUW' implemented in Julia file executable from a bash `comptimes.jl`. The script returns charts of the computional speed-up
of introduced algorithm vs other algorithms.
It has optional arguments: -m (Int) - cumulant's order, -n (vararg Int) - numbers of marginal variables, -T (vararg Int) - number of realistations
of random variable. Be cautious while $m > 4$, naive algorithm calculations may need a large time, up to $m!$ larger compared with an introduced algorithm, 
there may be a memory shortage since the comparison algorithms does not use a block structure and stores in a computer mamory a whole cumulant array of size $m^n$.

The script `gandata.jl` executable fram bash. They were generated form the t-multivariate distribution with $\nu = 14$ degrees of freedom, using `gandata.jl`.
There are $n = 4$ marginal variables and $T = 75000000$ realisations.
