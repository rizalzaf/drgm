using MathProgBase

include("types.jl")

## Computing marginals on tree
## See Wainwright MLSS 2012 Graphical Models and Message Passing tutorial
## http://yosinski.com/mlss12/media/slides/MLSS-2012-Wainwright-Graphical-Models-and-Message-Passing.pdf
function compute_marginal(T::Vector{Int}, EX_node::Matrix{Float64}, EX_edge::Array{Float64,3})
    nc, nn = size(EX_node)

    Z = zeros(Int, nc, nn)          # store argmax
    F = zeros(nc, nn)               # store max

    # upward (bottom up)
    ## computing message from child to parent
    M_up = zeros(nc, nn)                             # store messages from i to T[i]
    @inbounds for i = nn:-1:1
        # find children of i
        ch_ids = find(T .== i)
        
        for j = 1:nc                                 # each T[i]'s class
            msg = 0.0
            for k = 1:nc                             # each i's class
                m = EX_edge[j, k, i] * EX_node[k, i]
                for l in ch_ids                      # for each child of i
                    m *= M_up[k, l]
                end
                msg += m
            end

            M_up[j, i] = msg                         # store message
        end
    end

    # downward (top down)
    ## computing message from parent to child
    M_down = zeros(nc, nn)                           # store messages from T[i] to i
    M_down[:, 1] = 1.0                               # initial message from node 0 to 1
    @inbounds for i = 2:nn
        # find siblings of i
        sb_ids = find(T .== T[i])
        
        for j = 1:nc                                       # each i's class
            msg = 0.0
            for k = 1:nc                                   # each T[i]'s class
                m = EX_edge[k, j, i] * EX_node[k, T[i]]
                m *= M_down[k, T[i]]                       # message from T[T[i]] to T[i]
                for l in sb_ids                            # message from siblings to T[i]
                    if l != i
                        m *= M_up[k, l]                    # take from upward messages
                    end
                end
                msg += m
            end

            M_down[j, i] = msg                             # store message
        end
    end

    # compute node marginals
    rs = zeros(nc, nn)
    for i = 1:nn
        # find children of i
        ch_ids = find(T .== i)

        # un normalized
        r_un = EX_node[:, i] .* M_down[:, i]                    # multiplied by parent
        for j in ch_ids
            r_un .*= M_up[:, j]
        end

        rs[:, i] = r_un / sum(r_un)
    end

    # compute edge marginals
    Qs = zeros(nc, nc, nn)
    for i = 1:nn
        ti = T[i]                                   # parent of i

        i_ch_ids = find(T .== i)                    # children of i
        ti_ch_ids = find(T .== ti)                  # children of parent of i

        Q_un = zeros(nc, nc)                        # unnormalized
        for j = 1:nc
            for k = 1:nc
                v =  EX_node[k, i] * EX_edge[j, k, i]
                if i != 1
                    v *= EX_node[j, T[i]]
                    v *= M_down[j, T[i]]    # message from parent of T[i]
                end
                for l in ti_ch_ids
                    if l != i
                        v *= M_up[j, l]     # messages from children of T[i] except i
                    end
                end
                for l in i_ch_ids
                    v *= M_up[k, l]     # messages from children of i
                end
                Q_un[j, k] = v
            end
        end

        Qs[:,:,i] = Q_un / sum(Q_un)
    end

    # compute log Z
    Z = 0.0
    ch_ids = find(T .== 1)                    # children root nodes
    for i = 1:nc
        v = EX_node[i, 1] 
        for j in ch_ids
            v *= M_up[i, j]
        end

        Z += v
    end
    logZ = log(Z + 1e-6)

    return Qs, rs, logZ
end

function fg_tree_crf(theta_node::Matrix{Float64}, theta_edge::Array{Float64,3}, Ti::Vector{Int}, 
    Xi_node::Matrix{Float64}, Xi_edge::Matrix{Float64}, Yi::Vector{Int}, lambda::Float64)
    
    # theta_node : nf x nc
    # theta_edge : nf x nc x nc
    # Xi_node : nf x nn
    # Xi_edge : nf x nn         : first Xi_edge is zeros
    # PS_node : nc x nn
    # PS_edge : nc x nc x nn    : first PS_edge is zeros
    # Qs : nc x nc x nn    
    # rs : nc x nn    

    nc = size(theta_node, 2)
    nn = length(Ti)    # of nodes
    nf_node = size(Xi_node, 1)
    nf_edge = size(Xi_edge, 1)

    # potentials
    PS_node = theta_node' * Xi_node
    PS_edge = reshape(reshape(theta_edge, (nf_edge, nc^2))' * Xi_edge, (nc, nc, nn))

    # exp potentials
    EX_node = exp.(PS_node)
    EX_edge = exp.(PS_edge)

    # compute marginals
    Qs, rs, logZ = compute_marginal(Ti, EX_node, EX_edge)

    ## objective
    # empirical
    val = logZ                                      # normalization
    @inbounds for i = 1:nn
        val -= PS_node[Yi[i], i]                    # empirical node
    end
    @inbounds for i = 2:nn
        val -= PS_edge[Yi[Ti[i]], Yi[i], i]         # empirical edge
    end
    # regularization
    val += (lambda / 2) * (sum(theta_node .^ 2) + sum(theta_edge .^ 2))

    # gradient for theta_node
    # combine expected with empirical
    for i = 1:nn
        rs[Yi[i], i] -= 1.0
    end
    grad_node = Xi_node * rs'                        # expected - empirical
    grad_node .+= lambda .* theta_node

    # gradient for theta_edge
    # combine expected with empirical
    for i = 2:nn
        Qs[Yi[Ti[i]], Yi[i], i] -= 1.0
    end
    grad_edge = reshape(Xi_edge * reshape(Qs, (nc^2, nn))', (nf_edge, nc, nc))
    grad_edge .+= lambda .* theta_edge     # penalty

    return val, grad_node, grad_edge
end

function train_tree_crf(TS::Vector{Vector{Int}}, XS_node::Vector{<:AbstractMatrix{Float64}}, XS_edge::Vector{<:AbstractMatrix{Float64}}, 
    YS::Vector{Vector{Int}}, nc::Int, lambda::Float64;
    alg::Symbol=:SGD, step::Float64=1.0, ftol::Float64=1e-6, grtol::Float64=1e-6, 
    max_iter::Int=100_000, iter_check::Int=100, max_pass::Int=500, verbose::Bool=false)

    # TS :: Vector{Vector{Int}} 
    # trees : Format: array of int, id = nodes, val = parent, first id is the root
    # e.g. [0, 1, 1]  
    ## XS_node :: Vector{Matrix{Float64}}
    ## XS_edge :: Vector{Matrix{Float64}} : for the root, edge features is a zero matrix
    ## alg :: Symbol :SGD, :AdaGrad, or :AdaDelta : algorithm for stochastic optimization
    # nc : number of class
    
    ns = length(TS)                         # of samples
    # nc                                    # of class
    nf_node = size(XS_node[1], 1)           # of node feature
    nf_edge = size(XS_edge[1], 1)           # of edge feature

    # parameters. init with zero
    theta_node = rand(nf_node, nc) - 0.5
    theta_edge = rand(nf_edge, nc, nc) - 0.5
    
    # adagrad
    # rate is something you need to set beforehand
    rate = step
    square_g_node = zeros(nf_node, nc)       # for storing historical square of grads
    square_g_edge = zeros(nf_edge, nc, nc)   # for storing historical square of grads
    ϵ = 1e-8                            # for numerical stability

    # adadelta
    rho = 0.95                            # decay parameter   
    ϵ = 1e-6                              # for numerical stability
    acc_g_node = zeros(nf_node, nc)            # accumulate gradients
    acc_g_edge = zeros(nf_edge, nc, nc)        # accumulate gradients
    acc_update_node = zeros(nf_node, nc)       # accumulate updates
    acc_update_edge = zeros(nf_edge, nc, nc)   # accumulate updates

    f_prev = Inf
    iter = 1
    npass = 1
    need_exit = false
    while true

        idx = randperm(ns)
        
        for i in idx
            Ti = TS[i] :: Vector{Int}
            Xi_node = Array(XS_node[i]) :: Matrix{Float64}
            Xi_edge = Array(XS_edge[i]) :: Matrix{Float64}
            Yi = YS[i] :: Vector{Int}
            
            f, grad_node, grad_edge = fg_tree_crf(theta_node, theta_edge, Ti, Xi_node, Xi_edge, Yi, lambda)

            if alg == :AdaGrad  # adagrad
                square_g_node += grad_node .^ 2
                square_g_edge += grad_edge .^ 2

                theta_node = theta_node - (rate ./ (sqrt.(square_g_node) + ϵ)) .* grad_node
                theta_edge = theta_edge - (rate ./ (sqrt.(square_g_edge) + ϵ)) .* grad_edge
            
            elseif alg == :AdaDelta  #adadelta
                acc_g_node = rho * acc_g_node + ( (1. - rho) * grad_node .^ 2 )
                acc_g_edge = rho * acc_g_edge + ( (1. - rho) * grad_edge .^ 2 )

                delta_node = -(sqrt.(acc_update_node + ϵ) ./ sqrt.(acc_g_node + ϵ)) .* grad_node
                delta_edge = -(sqrt.(acc_update_edge + ϵ) ./ sqrt.(acc_g_edge + ϵ)) .* grad_edge

                acc_update_node = rho * acc_update_node + ( (1. - rho) * delta_node .^ 2 )
                acc_update_edge = rho * acc_update_edge + ( (1. - rho) * delta_edge .^ 2 )

                theta_node = theta_node + delta_node
                theta_edge = theta_edge + delta_edge

            else   # SGD
                theta_node = theta_node - step * grad_node
                theta_edge = theta_edge - step * grad_edge
            end

            # check progress every iter_check
            if iter % iter_check == 0
            
                if verbose 
                    println("iter : ", iter, ", pass : ", npass, ", f : ", f, 
                        ", abs grad node : ", mean(abs.(grad_node)), ", abs grad edge : ", mean(abs.(grad_edge))) 
                end

                if alg == :SGD
                    # discount step
                    step = step * 0.98
                end

                if iter >= max_iter
                    if verbose println("maximum iteration reached!!") end
                    need_exit = true
                    break
                end

                # if mean(abs.(grad_node)) < grtol && mean(abs.(grad_edge)) < grtol
                #     if verbose println("gradient breaks!!") end
                #     need_exit = true
                #     break
                # end

                # if abs(f_prev - f) < ftol
                #     if verbose println("function breaks!!") end
                #     need_exit = true
                #     break
                # end
                f_prev = f

            end

            iter = iter + 1
        end

        if npass >= max_pass
            if verbose println("maximum pass reached!!") end
            break
        end

        if need_exit
            break
        end

        npass = npass + 1
    end

    return CRFTreeModel(theta_node, theta_edge)

end

function predict_tree_crf(model::CRFTreeModel, TS::Vector{Vector{Int}}, XS_node::Vector{<:AbstractMatrix{Float64}}, XS_edge::Vector{<:AbstractMatrix{Float64}}, L_base::Symbol, LS_weights::Vector{Vector{Float64}})
    ## Bayes response prediction

    ## L_base : base loss
    ## LS_weights :: Vector{Vector{Float64}} : weight of loss for each node

    ns = length(TS)                         # of samples
    # get theta 
    theta_node = model.theta_node
    theta_edge = model.theta_edge

    nc = size(theta_node, 2)
    nf_node = size(XS_node[1], 1)
    nf_edge = size(XS_edge[1], 1)

    YS_pred = Vector{Vector{Int}}(ns)

    for i = 1:ns
        Ti = TS[i] :: Vector{Int}
        Xi_node = Array(XS_node[i]) :: Matrix{Float64}
        Xi_edge = Array(XS_edge[i]) :: Matrix{Float64}
        Li_weights = LS_weights[i] :: Vector{Float64}

        nn = length(Ti)
        nf_edge = size(Xi_edge, 1)

        # potentials
        PS_node = theta_node' * Xi_node
        PS_edge = reshape(reshape(theta_edge, (nf_edge, nc^2))' * Xi_edge, (nc, nc, nn))

        # exp potentials
        EX_node = exp.(PS_node)
        EX_edge = exp.(PS_edge)

        # get the node marginal probability
        _, rs, _ = compute_marginal(Ti, EX_node, EX_edge)

        Yi_pred = zeros(Int, nn)
        for j = 1:nn
            if L_base == :zeroone                   # weights doesn't matter
                Yi_pred[j] = indmax(rs[:, j])
            elseif L_base == :absolute
                expected_loss = zeros(nc)
                for k = 1:nc
                    v = 0.0
                    for l = 1:nc
                        v += abs(k - l) * rs[l, j]
                    end
                    expected_loss[k] = v
                end
                Yi_pred[j] = indmin(expected_loss)
            elseif L_base == :squared
                expected_loss = zeros(nc)
                for k = 1:nc
                    v = 0.0
                    for l = 1:nc
                        v += (k - l)^2 * rs[l, j]
                    end
                    expected_loss[k] = v
                end
                Yi_pred[j] = indmin(expected_loss)
            end
        end

        YS_pred[i] = Yi_pred

    end

    return YS_pred
end

function test_tree_crf(YS_pred::Vector{Vector{Int}}, YS::Vector{Vector{Int}}, L_base::Symbol, LS_weights::Vector{Vector{Float64}})

    ns = length(YS)                         # of samples
    v_loss = zeros(ns)
    for i = 1:ns
        Yi = YS[i]
        Yi_pred = YS_pred[i]
        Li_weights = LS_weights[i]

        nn = length(Yi)
        ls = 0.0    # loss
        for j = 1:nn 
            # weighted loss
            weight = Li_weights[j]
            if L_base == :zeroone
                ls += weight * Int(Yi_pred[j] != Yi[j])
            elseif L_base == :absolute
                ls += weight * abs(Yi_pred[j] - Yi[j])
            elseif L_base == :squared
                ls += weight * (Yi_pred[j] - Yi[j])^2
            end
        end

        ls = ls / sum(Li_weights)    # divide by sum of weights

        v_loss[i] = ls
    end

    avg_loss = mean(v_loss)

    return avg_loss, v_loss
end

function test_tree_crf(model::CRFTreeModel, TS::Vector{Vector{Int}}, XS_node::Vector{<:AbstractMatrix{Float64}}, XS_edge::Vector{<:AbstractMatrix{Float64}}, 
    YS::Vector{Vector{Int}}, L_base::Symbol, LS_weights::Vector{Vector{Float64}})

    # prediction
    YS_pred = predict_tree_gm(model, TS, XS_node, XS_edge, L_base, LS_weights)    

    return test_tree_gm(YS_pred, YS, L_base, LS_weights)
end

function nodewise_test_tree_crf(YS_pred::Vector{Vector{Int}}, YS::Vector{Vector{Int}}, L_base::Symbol, LS_weights::Vector{Vector{Float64}})

    ns = length(YS)                         # of samples
    sum_weight_all = 0.0
    loss = 0.0
    for i = 1:ns
        Yi = YS[i]
        Yi_pred = YS_pred[i]
        Li_weights = LS_weights[i]

        nn = length(Yi)
        ls = 0.0    # loss
        for j = 1:nn 
            # weighted loss
            weight = Li_weights[j]
            if L_base == :zeroone
                ls += weight * Int(Yi_pred[j] != Yi[j])
            elseif L_base == :absolute
                ls += weight * abs(Yi_pred[j] - Yi[j])
            elseif L_base == :squared
                ls += weight * (Yi_pred[j] - Yi[j])^2
            end
        end

        loss += ls
        sum_weight_all += sum(Li_weights)
    end

    avg_loss = loss / sum_weight_all
    
    return avg_loss
end
