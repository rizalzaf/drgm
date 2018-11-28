## argmax of f prediction
function predict_argmax_f(T::Vector{Int}, PS_node::Matrix{Float64}, PS_edge::Array{Float64,3})

    nc, nn = size(PS_node)

    Z = zeros(Int, nc, nn)          # store argmax
    F = zeros(nc, nn)               # store max

    # find max
    @inbounds for i = nn:-1:1
        ch_ids = find(T[i+1:end] .== i)
        if length(ch_ids) == 0          # leaf node
            for k = 1:nc                # each class
                candidate = zeros(nc)
                for l = 1:nc
                    candidate[l] = PS_node[l,i] + PS_edge[k,l,i] + PS_node[k,T[i]]
                end
                v, id = findmax(candidate)
                F[k,i] = v
                Z[k,i] = id
            end
        else                            # has childeren
            for k = 1:nc                # each class
                candidate = zeros(nc)
                for l = 1:nc
                    for j in ch_ids         
                        candidate[l] += F[l, i+j] - PS_node[l,i]
                    end
                    candidate[l] +=  PS_node[l,i] + PS_edge[k,l,i] + ( i != 1 ? PS_node[k,T[i]] : 0.0 )
                end
                v, id = findmax(candidate)
                F[k,i] = v
                Z[k,i] = id
            end
        end
    end

    # backtrack
    val = F[1,1]
    sol = zeros(Int, nn)
    sol[1] = Z[1,1]
    @inbounds for i = 2:nn
        sol[i] = Z[sol[T[i]],i]
    end
    
    return sol, val
end

