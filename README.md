# Cumulants.jl

Calculates Cummulant tensors of any order for multivariate data. 
Functions always return tensor or array of tensors in `SymmetricTensors` type. Raquires SymmetricTensors.jl "https://github.com/kdomino/SymmetricTensors.jl". To convert to array, run:

```julia
julia> convert(Array, data::SymmetricTensors{T})
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
julia> moment(data::Matrix{T}, order::Int, bls::Int = 2)
```

Returns a tensor of the moment of the given order of multivariate data.
For multivariate data takes matrix, where columns numerates marginal variables and rows 
numertates their realisations, bls is an optional Int that determines a size 
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
julia> cumulants(data::Matrix{T}, maxord::Int, bls::Int = 2)
```

Returns a vector of tensors of cumulants of order 2,3,...,maxord of multivariate data.
Input the same as for the moment function. Returns a vector of `SymmetricTensors`.

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


