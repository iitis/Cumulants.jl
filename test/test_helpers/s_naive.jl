#uses partitions and reccurence does not use blocks
"""
produces partitions of size <= 2 for recurrence
combination of lower cumulants
"""
function partitions2(array::Vector{Int})
    a = Vector{Vector{Int}}[]
    n = length(array)
    @inbounds for p in partitions(array)
      s = map(length, p)
        if !((1 in s) | (n in s))
            push!(a, p)
        end
    end
    return a
end


"""
no blocks, result has to be asign for all
indices permutation of output super symmetric tensor
"""
macro per(m, a, i)
    quote
      ind = [$i...]
        @inbounds for per in collect(permutations(ind))
            $m[per...] = $a
        end
    end
end

function permutations_output!{T<:AbstractFloat, N}(m::Array{T, N}, a::T, list::Vector{Int})
    @per(m, a, list)
end


"""
calculates a single element of moment tensor and
asign it to the moments tensor at each permutation of indices
(they are pyramidal for calculations only)
"""
function moment_element!{T<:AbstractFloat, N}(moment::Array{T, N}, ind::Vector{Int}, data::Matrix{T})
    temp = 1.
    @inbounds for i in ind
        temp  = temp.*data[:,i]
    end
    permutations_output!(moment, mean(temp), ind)
end


"""
evaluates pyramid indices for calculation
The method exploit super symmetry only for calculations
"""
function indice(n::Int, s::Int)
    ret = Vector{Int}[]
    @eval begin
        @nloops $n i x -> (x==$n)? (1:$s): (i_{x+1}:$s) begin
            ind = @ntuple $n x -> i_{$n-x+1}
            @inbounds push!($ret, [ind...])
        end
    end
    ret
end


"""
calculates nth moment

output n dims array
"""
function moment_n{T<:AbstractFloat}(data::Matrix{T}, n::Int)
    s = size(data,2)
    moment = zeros(fill(s,n)...)
    @inbounds for i in indice(n, s)
        moment_element!(moment, [i...], data)
    end
    moment
end


"""
element of the sum of products of lower cumulants

output float number
"""
function calculate_el{T<:AbstractString}(c::Dict{T ,Any}, list::Vector{Int})
    a = partitions2(list)
    sum = 0.
    @inbounds for k = 1:length(a)
        prod = 1.
        @inbounds for el in a[k]
            prod*= c["c"*"$(size(el,1))"][el...]
        end
        sum += prod
    end
    sum
end


"""
the sum of products of lower cumulants

output array
"""
function produ{T<:AbstractString}(c::Dict{T ,Any}, n::Int)
    s = size(c["c2"], 2)
    ret = zeros(fill(s,n)...)
    @inbounds for i in indice(n, s)
        permutations_output!(ret, calculate_el(c, [i...]), [i...])
    end
    ret
end


"""
calculate cumulants

imput: data for which cumulants are calculated, n - order of highest comulant

output: cumulants up to n (dict of arrays)
"""
function snaivecumulant{T<:AbstractFloat}(data::Matrix{T}, n::Int)
    data = center(data);
    c2 = Base.covm(data, 0, 1, false)
    cumulants = Dict("c2" => c2, "c3" => moment_n(data, 3));
    for k in 4:n
      merge!(cumulants, Dict("c$k" => moment_n(data, k) - produ(cumulants, k)))
    end
    return cumulants
end
