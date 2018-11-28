
function al_zo(ps::Vector, scale::Real=1.0)

    nc = length(ps)
    id = sortperm(ps, rev=true)    # sort, get the indices

    sum_ps = ps[id[1]]
    th = 1
    for j = 2:nc
        if (j-1) * ps[id[j]] + scale >= sum_ps      # if tie choose the one with more members
            sum_ps += ps[id[j]]
            th = j
        else
            break
        end
    end

    loss = (sum_ps + (th - 1) * scale) / th 

    r = zeros(nc)
    for i = 1:th
        r[id[i]] = 1 / th
    end

    return loss, r
end

function al_ord(ps::Vector, scale::Real=1.0)
    
    nc = length(ps)
  
    mi = ps[1] - scale
    mj = ps[1] + scale 
    id_i = 1
    id_j = 1
    for i = 2:nc
        v = ps[i] - scale * i
        if v >= mi              # if tie choose the one with colosest gap
            mi = v
            id_i = i
        end
        v = ps[i] + scale * i
        if v > mj               # if tie choose the one with colosest gap
            mj = v
            id_j = i
        end
    end
  
    loss = (mi + mj) / 2

    r = zeros(nc)
    r[id_i] = 0.5
    r[id_j] = 0.5
  
    return loss, r
end

function al_quad(ps::Vector, scale::Real=1.0)
    nc = length(ps)
  
    loss = -Inf
    best_i = 0
    best_j = 0
    best_k = 0
  
    for i = 1:nc
        for j = i+1:nc
            denom = 2(j-i)
            for k = i+1:j
                ai = 2(j-k) + 1
                aj = 2(k-i) - 1
                ls = ai * (ps[i] + scale * (k-i)^2) + aj * (ps[j] + scale * (j-k)^2)
                ls = ls / denom
            
                if ls > loss
                    loss = ls
                    best_i = i
                    best_j = j
                    best_k = k
                end        
            end      
        end
    end
  
    v1, id1 = findmax(ps)
    r = zeros(nc)
  
    if v1 > loss
        loss = v1
        r[id1] = 1.0
    else
        denom = 2(best_j-best_i)
        ai = 2(best_j-best_k) + 1
        aj = 2(best_k-best_i) - 1
        r[best_i] = ai / denom
        r[best_j] = aj / denom
    end
  
    return loss, r
  end
  

function solve_r(a::Vector{Float64}, L_base::Symbol, weight::Float64)

    # solve r using exact solver for zero-one and ordinal (absolute) loss
    if L_base == :zeroone
        val, r = al_zo(a, weight)
    elseif L_base == :absolute
        val, r = al_ord(a, weight)
    elseif L_base == :squared
        val, r = al_quad(a, weight)
    else
        error("Only accept :zeroone, :absolute, and :squared")
    end

    return val::Float64, r::Vector{Float64}
end

function solve_Q_given_U(U::Matrix{Float64}, T::Vector{Int}, L_base::Symbol, L_weights::Vector{Float64}, 
    PS_node::Matrix{Float64}, PS_edge::Array{Float64,3})
    
    # PS_node : (nc,nn) | PS_edge : (nc,nc,nn) | U : (nc,nn-1) | 
    # Q : (nc,nc,nn) | T : (nn)
    # first PS_edge : all zeros

    # get sizes
    nc = size(PS_node, 1)
    nn = length(T)

    # storage
    val = 0.0
    Qs = zeros(nc,nc,nn)
    rs = zeros(nc,nn)

    As = zeros(nc,nc,nn)
    as = zeros(nc,nn)

    # solve decomposed inner min max given lagrange variable 
    @inbounds for k = 1:nn
        # construct A
        A = PS_edge[:,:,k]
        A = A .+ PS_node[:,k]'

        if k > 1; A = A .- U[:,k-1]; end
        for i in find(T .== k)      # find if k is parents of any node
            A = A .+ U[:,i-1]'
        end

        As[:,:,k] = A

        a, idm = findmax(A, 1)
        v, r = solve_r(vec(a), L_base, L_weights[k])

        as[:,k] = a

        # store result
        rs[:,k] = r
        Q = zeros(nc, nc)
        Q[idm] = r
        Qs[:,:,k] = Q
        val += v
    end

    return val, Qs, rs, As, as
end


function U_objective(U::Matrix{Float64}, T::Vector{Int}, L_base::Symbol, L_weights::Vector{Float64},
    PS_node::Matrix{Float64}, PS_edge::Array{Float64,3})

    val, _, _, _ = solve_Q_given_U(U, T, L_base, L_weights, PS_node, PS_edge)
    return val
end

function U_grad!(gr::Vector{Float64}, U::Matrix{Float64}, T::Vector{Int}, L_base::Symbol, L_weights::Vector{Float64},
    PS_node::Matrix{Float64}, PS_edge::Array{Float64,3})

    # get sizes
    nn = length(T)
    # get Q matrices
    val, Qs, _ = solve_Q_given_U(U, T, L_base, L_weights, PS_node, PS_edge)

    # gradient
    dU = zeros(size(U))
    @inbounds for i = 1:nn-1
        dU[:,i] = vec(sum(Qs[:,:,T[i+1]], 1)) - vec(sum(Qs[:,:,i+1], 2))
    end

    gr[:] = vec(dU)
end

function solve_U(T::Vector{Int}, L_base::Symbol, L_weights::Vector{Float64},
    PS_node::Matrix{Float64}, PS_edge::Array{Float64,3};
    g_tol::Real = 1e-5, f_tol = 1e-5)

    # get sizes
    nc = size(PS_node, 1)
    nn = length(T)
    m = nn - 1

    # starting
    u = 2rand(nc * m) - 1

    res = Optim.optimize( 
            x -> U_objective(reshape(x, (nc, m)), T, L_base, L_weights, PS_node, PS_edge),
            (g, x) -> U_grad!(g, reshape(x, (nc, m)), T, L_base, L_weights, PS_node, PS_edge), 
            u,
            BFGS(linesearch = LineSearches.HagerZhang()),
            Optim.Options(g_tol = g_tol,
                f_tol = f_tol,
                iterations = 100,
                store_trace = false,
                show_trace =  false) #true)
            )

    u = Optim.minimizer(res) :: Vector{Float64}
    U = reshape(u, nc, m)
    val = Optim.minimum(res) :: Float64

    return val, U
end

function max_emd(D::Matrix, ps::Vector, pt::Vector, lp_solver::MathProgBase.AbstractMathProgSolver)
    # maximum earth mover distance (instead of the standard minimum)
    # D : distance
    # ps : source
    # pt : target

    # solve r using linear programming
    # minimimize instead of maximize, hence, all signs are flipped

    # avoid numerical instability that may cause infeasible 
    ps = ps ./ sum(ps)
    pt = pt ./ sum(pt)

    nc = length(ps)
    nv = nc^2

    # objective
    obj = -vec(D)

    # constraints
    ncs = 2nc
    Acs = spzeros(ncs, nv)
    bcs = zeros(ncs)
    sense = Vector{Char}(ncs)
    @inbounds for it = 1:nc
        for j = 1:nc
            id = sub2ind((nc, nc), it, j)
            Acs[it, id] = 1.0
        end
        bcs[it] = ps[it]
        sense[it] = '='
    end
    @inbounds for it = 1:nc
        for j = 1:nc
            id = sub2ind((nc, nc), j, it)
            Acs[nc + it, id] = 1.0
        end
        bcs[nc + it] = pt[it]
        sense[nc + it] = '='
    end

    # solve qp
    solution = linprog(obj, Acs, sense, bcs, 0.0, Inf, lp_solver)
    q = solution.sol :: Vector{Float64}
    # use max(0, x) to address numerical instability
    q = max.(0.0, q)

    # get the matrix
    Q = reshape(q, (nc, nc))

    return Q
end

function find_Q_given_r(rs::Matrix{Float64}, T::Vector{Int},  
    PS_edge::Array{Float64,3}, lp_solver::MathProgBase.AbstractMathProgSolver)
    
    # PS_node : (nc,nn) | PS_edge : (nc,nc,nn) | U : (nc,nn-1) | 
    # Q : (nc,nc,nn) | T : (nn)
    # first PS_edge : all zeros

    # get sizes
    nc = size(rs, 1)
    nn = length(T)

    # storage
    Qs = zeros(nc,nc,nn)
    
    Qs[1,:,1] = rs[:,1]                    # first Q, Q_{0,1} => dummy vars
    @inbounds for k = 2:nn
        ps = rs[:,T[k]]                  # source
        pt = rs[:,k]                     # target
        D = PS_edge[:,:,k]               # cost

        Q = max_emd(D, ps, pt, lp_solver)
        Qs[:,:,k] = Q
    end

    return Qs
end

function objective_Q(Qs::Array{Float64,3}, rs::Matrix{Float64}, T::Vector{Int}, L_base::Symbol, L_weights::Vector{Float64},
    PS_node::Matrix{Float64}, PS_edge::Array{Float64,3})

    nn = length(T)
    nc = size(PS_node, 1)

    val = sum(rs .* PS_node) + sum(Qs .* PS_edge)   
    for i = 1:nn
        weight = L_weights[i]
        minv = Inf
        for j = 1:nc
            if L_base == :zeroone
                ls = ones(nc)
                ls[j] = 0.0
            elseif L_base == :absolute
                ls = [abs(k - j) for k = 1:nc]
            elseif L_base == :squared
                ls = [(k - j)^2 for k = 1:nc]
            else
                error("Only accept :zeroone, :absolute and squared")
            end
            ls = ls * weight

            v = dot(ls, rs[:,i])
            if v < minv
                minv = v
            end
        end

        val += minv
    end

    return val::Float64
end

function solve_minimax_Q(T::Vector{Int}, L_base::Symbol, L_weights::Vector{Float64},
    PS_node::Matrix{Float64}, PS_edge::Array{Float64,3},
    lp_solver::MathProgBase.AbstractMathProgSolver)

    # solve lagrange dual to get U
    _, best_U = solve_U(T, L_base, L_weights, PS_node, PS_edge)

    # given U, get r. Since the function is smooth over r, we will get the optimal primal r 
    _, _, best_r = solve_Q_given_U(best_U, T, L_base, L_weights, PS_node, PS_edge)

    # to get Q, find Q using LP, or maximum earth mover distance 
    best_Q = find_Q_given_r(best_r, T, PS_edge, lp_solver)

    return best_Q, best_r
    
end

