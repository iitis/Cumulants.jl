using Combinatorics
using DataStructures

# implementation of J. De Leeuw Multivariate Cumulants in R (2012)

function outer(a::Vector{T}, b::Vector{T}) where T<: AbstractFloat
    sa = size(a,1)
    sb = size(b,1)
    z = zeros(T, sa*sb)
    k = 1
    for i = 1:sb
        for j = 1:sa
            @inbounds z[k] = a[j]*b[i]
            k += 1
        end
    end
    return z
end

function setparts(n)
    ret = zeros(Int, n, length(partitions(collect(1:n))))
    for (i, partition) in enumerate(partitions(collect(1:n)))
        group = 1
        for p in partition
            for j in p
                ret[j,i] = group
            end
            group += 1
        end
    end
    ret
end

function raw_moments_upto_p(x, p=4)
    n, m = size(x)
    if p==1
        return vcat(1, mean(x, axis=2))
    end
    y = zeros(eltype(x), (m+1)^p)
    for i in 1:n
        xi = vcat(1, x[i,:])
        #z = kron([xi for _ in 1:p]...)
        z = xi
        for j in 2:p
          z = outer(xi, z)
        end
        y += z
    end
    reshape(y, repeat([m+1],p)...)./n
end

function cumulants_from_raw_moments(raw)
        dimr = size(raw)
        nvar::Int64 = dimr[1]
        cumu = zeros(eltype(raw), dimr...)
        nele = prod(dimr)
        ldim = length(dimr)
        spp = Array{Array{Int64,2}}(undef, ldim)
        qpp::Array{Int64} = Array{Int64}(undef, ldim)
        rpp = Array{Any}(undef, ldim)

        function one_cumulant_from_raw_moments(jnd, raw)
            jnd = [jnd[find(jnd.!=1)]...] - 1
            nnd = length(jnd)
            ndr::Int64 = size(raw)[1]
            nrt = length(size(raw))
            raw = rpp[nnd]
            nvar = ndr - 1
            nraw = max(1, length(size(raw)))
            sp = spp[nraw]
            _, nbell = size(sp)
            sterm = 0.0
            for i in 1:nbell
                ind = sp[:, i]
                und = unique(ind)
                term = qpp[length(und)]
                for j in und
                    knd = jnd[find(ind.==j)] + 1
                    lnd =  vcat(knd, repeat([1], nraw - length(knd)))
                    term *= raw[lnd...]
                end
            sterm +=  term
            end
            return sterm
        end

        for i in 1:ldim
            spp[i] = setparts(i)
            qpp[i] = factorial(i)
            if mod(i,2)==1
                qpp[i] = -qpp[i]
            end
            rpp[i] = raw[hcat([collect(1:nvar) for i in 1:i])...]
        end
        qpp = vcat(1, qpp)
        for i in 2:nele
            ind = ind2sub(dimr, i)
            cumu[i] = one_cumulant_from_raw_moments(ind, raw)
        end
        return cumu
end

function cumulants_upto_p(x, p = 4)
    return cumulants_from_raw_moments(raw_moments_upto_p(x, p))
end

function first_four_cumulants(x)
    cumu = cumulants_upto_p(x)
    nsel = 2:size(cumu)[1]
    OrderedDict(:c1 => cumu[1, 1, 1, nsel],
                :c2 => cumu[1, 1, nsel, nsel],
                :c3 => cumu[1, nsel, nsel, nsel],
                :c4 => cumu[nsel, nsel, nsel, nsel]
                )
end
