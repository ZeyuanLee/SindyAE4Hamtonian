# using only forwarddiff, for multi orbit #

using Pkg
# Pkg.activate(".")
# include("model_simplechain.jl")

using GeometricMachineLearning:_optimization_step!,NeuralNetworkParameters
using DelimitedFiles
using Statistics
using ForwardDiff
using LinearAlgebra


# setting data #
const dt = 0.01
# filename = "exbl_rp.txt"
# zori = readdlm(filename)
# inputori = collect(transpose(zori))
files = ["orbit1.txt", "orbit2.txt", "orbit3.txt", "orbit4.txt"]
tra_ori = [collect(transpose(readdlm(file))) for file in files]
# TODO: try to use JLD2 or HDF5 to load data


# all_data_combined = hcat(tra_ori)
# input_mean = mean(all_data_combined, dims=2)
# input_std = std(all_data_combined, dims=2)

# nomalization #
# input = (inputori .- input_mean) ./ (input_std .+ 1e-8) # after normalized
# input = [(traj .- input_mean) ./ (input_std .+ 1e-8) for traj in tra_ori]
input = tra_ori

# setting model #
N_samples = size(input[1],2)
N_features = size(input[1],1) # for ex 4
output = zeros(N_features, N_samples)
N_dim = N_features ÷ 2
sindy_arch = PRPModel(N_features)
# nn = NeuralNetwork(Chain(Dense(2,10,tanh),Dense(10,10,tanh),Dense(10,4,tanh),# encoder  
                    # Chain(sindy_arch).layers..., # Nested Sindy 
                    # Dense(1,10,tanh),Dense(10,2,identity))) # decoder
nn = NeuralNetwork(Chain(Chain(sindy_arch).layers...,)) # only sindy layer

# tem_ps = (nn.params.L1,nn.params.L2,nn.params.L3,nn.params.L4,nn.params.L5,nn.params.L6)
#AbstractNeuralNetworks.Chain(nn.model.layers[1:6]...)(rand(2),tem_ps)

# ★★★ パラメータの初期化を上書き ★★★
# これにより、モデルがより「敏感」になり、大きな勾配を生成しやすくなる
println("L2の重みを新しい乱数で再初期化します...")
W_shape = size(nn.params.L2.W)
nn.params.L2.W .= 0.0 # 平均0, 標準偏差1の乱数で初期化
# nn.params.L2.W[3] = abs(nn.params.L2.W[3])
# nn.params.L2.W[6] = abs(nn.params.L2.W[6])
# nn.params.L2.W[8] = -abs(nn.params.L2.W[8])
# nn.params.L2.W[10] = abs(nn.params.L2.W[10])
# nn.params.L2.W[11] = -abs(nn.params.L2.W[11])

nn.params.L2.W[3] = 0.5
nn.params.L2.W[6] = 0.5
nn.params.L2.W[8] = -2.0
nn.params.L2.W[10] = 2.0
nn.params.L2.W[11] = -1.0
# learning setting #
# const batch_size = 50
const n_epochs = 3000
opt1 = GeometricMachineLearning.Optimizer(AdamOptimizer(1e-2), nn)
opt2 = GeometricMachineLearning.Optimizer(AdamOptimizer(1e-3), nn)
λ = GlobalSection(nn.params)


# extract weights in L2 layer which are parameters #
initial_params_vec = vec(nn.params.L2.W) 
# "vec" tranforms matrix into vector #

# remember the size of the weights #
original_shape = size(nn.params.L2.W)


function full_loss(p_vec::AbstractVector, all_tra::Vector)
    total_loss = 0.0

    for traj in all_tra
        total_loss += mse_loss(p_vec, traj)
    end

    return total_loss/length(all_tra)
end

function mse_loss(p_vec, traj)
    # 1. restore the parameters #
    W_matrix = reshape(p_vec, original_shape) # do not have to reshape?
    current_ps = (L1 = NamedTuple(), L2 = (W = W_matrix,))

    current_batch = traj
    z_input = current_batch[:, 1:end-1]
    dz_dt_true = (current_batch[:, 2:end] - current_batch[:, 1:end-1]) / dt
    num_samples = size(z_input, 2)

    # type of "p_vec" (Float64 or ForwardDiff.Dual) #
    T = eltype(p_vec)
    dz_dt_pred = Matrix{T}(undef, N_features, num_samples)

    for j in 1:num_samples
        z_sample = z_input[:, j]
        hamiltonian_func(vec_in) = nn(reshape(vec_in, :, 1), current_ps)[1, 1]
        grad_H = ForwardDiff.gradient(hamiltonian_func, z_sample)
        
        dH_dq = grad_H[1:N_dim]
        dH_dp = grad_H[(N_dim + 1):end]
        dz_dt_pred[:, j] = vcat(dH_dp, -dH_dq)
    end

    # though "dz_dt_pred" is dual matrix, it calculate properly "
    return mean(abs2, dz_dt_pred - dz_dt_true) #+ 1e-3* norm(W_matrix)
end


function SINDyAE_optimization_step!(o::Optimizer, λY::NamedTuple, ps::NeuralNetworkParameters, dx::Union{NamedTuple, NeuralNetworkParameters})
    @assert keys(o.cache) == keys(λY) == keys(ps) == keys(dx)
    o.step += 1
    for key in keys(o.cache)
        cache = o.cache[key]
        λY_temp = λY[key]
        ps_temp = ps[key]
        dx_temp = dx[key]
        if dx_temp == nothing
            continue
        end
        _optimization_step!(o, λY_temp, ps_temp, cache, dx_temp)
    end
end


# main part #
err = zeros(n_epochs)
p_vec = initial_params_vec

println("\n--- 学習開始 ---")


for ep in 1:1000    
    loss_grad(p_vec) = full_loss(p_vec, input)
    @show full_loss(p_vec, input)
    gs_vec = ForwardDiff.gradient(loss_grad, p_vec)
    
    # gs_vec -> NamedTuple #
    gs_matrix = reshape(gs_vec, original_shape)
    gs_namedtuple = (L1 = NamedTuple(), L2 = (W = gs_matrix,))
    
    # revew the parameters #
    if ep == 1
        SINDyAE_optimization_step!(opt1, λ, nn.params, gs_namedtuple)
    elseif err[ep-1] < 2.2
        SINDyAE_optimization_step!(opt1, λ, nn.params, gs_namedtuple)
    else
        SINDyAE_optimization_step!(opt2, λ, nn.params, gs_namedtuple)
    end
    
    # nn.params -> vector #
    global p_vec = vec(nn.params.L2.W)
    nn.params.L2.W .-= 1e-3 * nn.params.L2.W # update the parameters
    nn.params.L2.W[abs.(nn.params.L2.W).< 0.1] .= 0.0

    loss = full_loss(p_vec, input)
    err[ep] = loss
    println("err=", err[ep])
    println("nn.params.L2.W=", nn.params.L2.W)
end

println("p_vec=", p_vec)
println("\n--- 学習完了 ---")
println("最終的な損失: ", err[end])
