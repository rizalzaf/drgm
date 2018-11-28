using MAT

function prepare_emotion_data(matfile::String, weighted::Bool)

    D = matread(matfile)
    data = D["data"]

    ns = length(data)

    # output
    TS = Vector{Vector{Int}}(ns)
    XS = Vector{Matrix{Float64}}(ns)
    YS = Vector{Vector{Int}}(ns)
    LVS = Vector{Vector{Int}}(ns)
    ZS = Vector{Int}(ns)
    LS_weights = Vector{Vector{Float64}}(ns)

    for i = 1:ns
        X = data[i]["X"]
        Z = round(Int, data[i]["Y"])
        Y = vec(round.(data[i]["H"]))

        nn = length(Y)
        T = [j - 1 for j = 1:nn]
        LV = collect(1:nn)
        if weighted
            L_weights = LV / 1.0
        else
            L_weights = ones(nn)
        end

        TS[i] = T
        XS[i] = X
        YS[i] = Y
        LVS[i] = LV
        ZS[i] = Z
        LS_weights[i] = L_weights
    end
    
    nc = maximum(maximum.(YS))
    nz = maximum(ZS)

    # construct features
    XS_node = Vector{Matrix{Float64}}(ns) 
    XS_edge = Vector{Matrix{Float64}}(ns)
    nf0 = size(XS[1], 1)
    nf1 = nf0 + 1
    nf_node = nz * nf1
    nf_edge = nz * nf1    

    # nodes and edges
    for i = 1:ns
        nn = length(TS[i])
        z = ZS[i]
        # node potentials
        X_node = zeros(nf_node, nn)
        for j = 1:nn
            X_node[(z-1)*nf1+1 : (z-1)*nf1+nf0, j] = XS[i][:,j]
            X_node[z*nf1, j] = 1.0
        end
        XS_node[i] = X_node

        # edge potentials
        X_edge = zeros(nf_edge, nn)
        for j = 2:nn        # start from 2nd node. first one for root, with all zeros
            X_edge[(z-1)*nf1+1 : (z-1)*nf1+nf0, j] = abs.(XS[i][:,j] - XS[i][:,TS[i][j]])
            X_edge[z*nf1, j] = 1.0
        end
        XS_edge[i] = X_edge
    end

    return TS, XS_node, XS_edge, YS, LS_weights
end
