using MathProgBase

include("types.jl")
include("solve_minimax_train.jl")
include("adv_argmax_f.jl")

function fg_tree_gm(theta_node::Matrix{Float64}, theta_edge::Array{Float64,3}, Ti::Vector{Int}, 
    Xi_node::Matrix{Float64}, Xi_edge::Matrix{Float64}, Yi::Vector{Int}, 
    L_base::Symbol, Li_weights::Vector{Float64}, lambda::Float64,
    lp_solver::MathProgBase.AbstractMathProgSolver)
    
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

    PS_node = theta_node' * Xi_node
    PS_edge = reshape(reshape(theta_edge, (nf_edge, nc^2))' * Xi_edge, (nc, nc, nn))

    normalized_weights = Li_weights ./ sum(Li_weights)

    Qs, rs = solve_minimax_Q(Ti, L_base, normalized_weights, PS_node, PS_edge, lp_solver)
    # objective
    val = objective_Q(Qs, rs, Ti, L_base, normalized_weights, PS_node, PS_edge)
    @inbounds for i = 1:nn
        val -= PS_node[Yi[i], i]                    # empirical node
    end
    @inbounds for i = 2:nn
        val -= PS_edge[Yi[Ti[i]], Yi[i], i]         # empirical edge
    end

    # regularization
    val = val + (lambda / 2) * (sum(theta_node .^ 2) + sum(theta_edge .^ 2))

    # gradient for theta_node
    # combine adversary with empirical
    for i = 1:nn
        rs[Yi[i], i] -= 1.0
    end
    grad_node = Xi_node * rs'                        # adversary - empirical
    grad_node .+= lambda .* theta_node
    
    # gradient for theta_edge
    # combine adversary with empirical
    for i = 2:nn
        Qs[Yi[Ti[i]], Yi[i], i] -= 1.0
    end
    grad_edge = reshape(Xi_edge * reshape(Qs, (nc^2, nn))', (nf_edge, nc, nc))
    grad_edge .+= lambda .* theta_edge     # penalty

    return val, grad_node, grad_edge
end

function train_tree_gm(TS::Vector{Vector{Int}}, XS_node::Vector{<:AbstractMatrix{Float64}}, XS_edge::Vector{<:AbstractMatrix{Float64}}, 
    YS::Vector{Vector{Int}}, L_base::Symbol, LS_weights::Vector{Vector{Float64}}, nc::Int, lambda::Float64,
    lp_solver::MathProgBase.AbstractMathProgSolver;
    alg::Symbol=:SGD, step::Float64=1.0, ftol::Float64=1e-6, grtol::Float64=1e-6, 
    max_iter::Int=100_000, iter_check::Int=100, discount_check::Int=200, max_pass::Int=500, verbose::Bool=false)

    # TS :: Vector{Vector{Int}} 
    # trees : Format: array of int, id = nodes, val = parent, first id is the root
    # e.g. [0, 1, 1]  
    ## XS_node :: Vector{Matrix{Float64}}
    ## XS_edge :: Vector{Matrix{Float64}} : for the root, edge features is a zero matrix
    ## L_base : base loss
    ## LS_weights :: Vector{Vector{Float64}} : weight of loss for each node
    ## alg :: Symbol :SGD, :AdaGrad, or :AdaDelta : algorithm for stochastic optimization
    # nc : number of class
    
    ns = length(TS)                         # of samples
    # nc                                    # of class
    nf_node = size(XS_node[1], 1)           # of node feature
    nf_edge = size(XS_edge[1], 1)           # of edge feature

    # parameters. init with zero
    theta_node = zeros(nf_node, nc) - 0.5
    theta_edge = zeros(nf_edge, nc, nc) - 0.5
    
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
    tic()
    while true

        idx = randperm(ns)
        
        for i in idx
            Ti = TS[i] :: Vector{Int}
            Xi_node = Array(XS_node[i]) :: Matrix{Float64}
            Xi_edge = Array(XS_edge[i]) :: Matrix{Float64}
            Yi = YS[i] :: Vector{Int}
            Li_weights = LS_weights[i] :: Vector{Float64}

            f, grad_node, grad_edge = fg_tree_gm(theta_node, theta_edge, Ti, Xi_node, Xi_edge, Yi, L_base, Li_weights, lambda, lp_solver)

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

                toc()
                tic()
            end

            if alg == :SGD && iter % discount_check == 0
                # discount step
                step = step * 0.98
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

    toc()
    return AdvTreeModel(theta_node, theta_edge)

end

function predict_tree_gm(model::AdvTreeModel, TS::Vector{Vector{Int}}, XS_node::Vector{<:AbstractMatrix{Float64}}, XS_edge::Vector{<:AbstractMatrix{Float64}})
    
    ns = length(TS)                         # of samples
    # get theta 
    theta_node = model.theta_node
    theta_edge = model.theta_edge
    nc = size(theta_node, 2)

    nc = size(theta_node, 2)
    YS_pred = Vector{Vector{Int}}(ns)

    for i = 1:ns
        Ti = TS[i] :: Vector{Int}
        Xi_node = Array(XS_node[i]) :: Matrix{Float64}
        Xi_edge = Array(XS_edge[i]) :: Matrix{Float64}

        nn = length(Ti)
        nf_edge = size(Xi_edge, 1)

        PS_node = theta_node' * Xi_node
        PS_edge = reshape(reshape(theta_edge, (nf_edge, nc^2))' * Xi_edge, (nc, nc, nn))

        Yi_pred, _ = predict_argmax_f(Ti, PS_node, PS_edge)

        YS_pred[i] = Yi_pred

    end

    return YS_pred
end


function test_tree_gm(YS_pred::Vector{Vector{Int}}, YS::Vector{Vector{Int}}, L_base::Symbol, LS_weights::Vector{Vector{Float64}})

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

function test_tree_gm(model::AdvTreeModel, TS::Vector{Vector{Int}}, XS_node::Vector{<:AbstractMatrix{Float64}}, XS_edge::Vector{<:AbstractMatrix{Float64}}, 
    YS::Vector{Vector{Int}}, L_base::Symbol, LS_weights::Vector{Vector{Float64}})

    # prediction
    YS_pred = predict_tree_gm(model, TS, XS_node, XS_edge)    

    return test_tree_gm(YS_pred, YS, L_base, LS_weights)
end

function nodewise_test_tree_gm(YS_pred::Vector{Vector{Int}}, YS::Vector{Vector{Int}}, L_base::Symbol, LS_weights::Vector{Vector{Float64}})

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

