using HDF5, Clp
using Optim
using MathProgBase

include("adv_tree_gm.jl")
include("emotion/preprocess.jl")

# data file
matfile = "emotion/ck.mat"
idsfile = "emotion/ids_random.mat"

info = "ADV EMOTION "

# loss metrics
loss = :zeroone
# loss = :absolute
# loss = :squared
weighted = true

@show loss
println(info)

# learning params
# alg = :SGD
alg = :AdaGrad
# alg = :AdaDelta
step = 0.1
nc = 3

# lp solver
lp_solver = ClpSolver(LogLevel = 0)

# load and prepare data
# original data
TS_orig, XS_node_orig, XS_edge_orig, YS_orig, LS_weights_orig = prepare_emotion_data(matfile, weighted)

# random split
ids_random = matread(idsfile)["ids"]

# use first split for CV
ids = ids_random[:, 1]
TS = TS_orig[ids]
XS_node = XS_node_orig[ids]
XS_edge = XS_edge_orig[ids]
YS = YS_orig[ids]
LS_weights = LS_weights_orig[ids]

# number of train vs test
n_train = 120
n_test = 47

# train & test
TS_train, XS_node_train, XS_edge_train, YS_train, LS_weights_train = TS[1:n_train], XS_node[1:n_train], XS_edge[1:n_train], YS[1:n_train], LS_weights[1:n_train]
TS_test, XS_node_test, XS_edge_test, YS_test, LS_weights_test = TS[n_train+1:end], XS_node[n_train+1:end], XS_edge[n_train+1:end], YS[n_train+1:end], LS_weights[n_train+1:end]

## cross validation setup

# split training data into tr and val set
n_tr = round(Int, 0.7 * n_train)
n_val = n_train - n_tr

TS_tr, XS_node_tr, XS_edge_tr, YS_tr, LS_weights_tr = TS[1:n_tr], XS_node[1:n_tr], XS_edge[1:n_tr], YS[1:n_tr], LS_weights[1:n_tr]
TS_val, XS_node_val, XS_edge_val, YS_val, LS_weights_val = TS[n_tr+1:end], XS_node[n_tr+1:end], XS_edge[n_tr+1:end], YS[n_tr+1:end], LS_weights[n_tr+1:end]

## cross validation
lambdas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 0.0]

nl = length(lambdas)
cv_avg_loss = zeros(nl)

for i = 1:nl
    lm = lambdas[i]
    println("CV training, lamda = ", lm)

    model = train_tree_gm(TS_tr, XS_node_tr, XS_edge_tr, YS_tr, loss, LS_weights_tr, nc, lm, lp_solver,  
            alg = alg, step = step, verbose = true, iter_check = 500, max_iter=1_000_000, max_pass = 50)

    YS_pred = predict_tree_gm(model, TS_val, XS_node_val, XS_edge_val)
    avg_loss, v_loss = test_tree_gm(YS_pred, YS_val, loss, LS_weights_val)
    # avg_loss = nodewise_test_tree_gm(YS_pred, YS_val, loss, LS_weights_val)

    cv_avg_loss[i] = avg_loss
end

best_i = indmin(cv_avg_loss)
best_lambda = lambdas[best_i]

println("Evaluation : best lambda = ", best_lambda)

v_avg_loss = zeros(20)
v_nodewise_avg_loss = zeros(20)
v_v_loss = Vector{Vector{Float64}}(20)
v_model = Vector{AdvTreeModel}(20)

## 20 splits of datasets
for i = 1:20

    # get the split index
    ids = ids_random[:, i]

    # get the random index permutation
    TS = TS_orig[ids]
    XS_node = XS_node_orig[ids]
    XS_edge = XS_edge_orig[ids]
    YS = YS_orig[ids]
    LS_weights = LS_weights_orig[ids]

    # number of train vs test
    n_train = 120
    n_test = 47

    # get train & test split
    TS_train, XS_node_train, XS_edge_train, YS_train, LS_weights_train = TS[1:n_train], XS_node[1:n_train], XS_edge[1:n_train], YS[1:n_train], LS_weights[1:n_train]
    TS_test, XS_node_test, XS_edge_test, YS_test, LS_weights_test = TS[n_train+1:end], XS_node[n_train+1:end], XS_edge[n_train+1:end], YS[n_train+1:end], LS_weights[n_train+1:end]

    ## Train full using best_lambda
    model = train_tree_gm(TS_train, XS_node_train, XS_edge_train, YS_train, loss, LS_weights_train, nc, best_lambda, lp_solver,  
            alg = alg, step = step, verbose = true, iter_check = 500, max_iter=1_000_000, max_pass = 100)

    YS_pred = predict_tree_gm(model, TS_test, XS_node_test, XS_edge_test)

    avg_loss, v_loss = test_tree_gm(YS_pred, YS_test, loss, LS_weights_test)
    nodewise_avg_loss = nodewise_test_tree_gm(YS_pred, YS_test, loss, LS_weights_test)

    v_avg_loss[i] = avg_loss
    v_nodewise_avg_loss[i] = nodewise_avg_loss
    v_v_loss[i] = v_loss
    v_model[i] = model

    println("Evaluation, split ", i, " : avg loss = ", avg_loss)

end

avg_avg_loss = mean(v_avg_loss)
std_avg_loss = std(v_avg_loss)

avg_nodewise_avg_loss = mean(v_nodewise_avg_loss)
std_nodewise_avg_loss = std(v_nodewise_avg_loss)

println(info)

@show best_lambda
@show loss
@show weighted
@show alg
@show avg_avg_loss
@show std_avg_loss
@show avg_nodewise_avg_loss
@show std_nodewise_avg_loss

# save
# write results to HDF5 file
fid = h5open("results/ADV-emotion-" * string(loss) * "-w" * string(weighted) * ".h5", "w")
write(fid, "best_lambda", best_lambda)
write(fid, "v_avg_loss", v_avg_loss)
write(fid, "v_nodewise_avg_loss", v_nodewise_avg_loss)
write(fid, "cv_avg_loss", cv_avg_loss)          # cross validation
for i = 1:20
    write(fid, "v_loss/$i", v_v_loss[i])
    write(fid, "theta_node/$i", v_model[i].theta_node)
    write(fid, "theta_edge/$i", v_model[i].theta_edge)
end
close(fid)
