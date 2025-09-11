# using only forwarddiff, for exb, for multi data #
# data from implicit mid point in program file #

using Pkg
Pkg.activate(".")
include("model_simplechain.jl")

using GeometricMachineLearning:_optimization_step!,NeuralNetworkParameters
using DelimitedFiles
using Statistics
using ForwardDiff
using LinearAlgebra
using Functors

# setting data #
const dt = 0.05
files = ["./exb1.txt", "./exb2.txt", "./exb3.txt", "./exb4.txt"]
input = [collect(transpose(readdlm(file))) for file in files]

# setting model #
N_samples = size(input[1],2)
N_features = size(input[1],1) # for ex 4
output = zeros(N_features, N_samples)
N_dim = N_features ÷ 2
arch = PRPModel(N_features)
# nn = NeuralNetwork(Chain(Dense(6,10,tanh),Dense(10,10,tanh),Dense(10,4,tanh),# encoder 
#                     Chain(arch).layers..., # Nested Sindy 
#                     Dense(4,10,tanh),Dense(10,6,identity))) # decoder
nn = NeuralNetwork(Chain(Chain(arch).layers...,)) # only sindy layer

W_shape = size(nn.params.L2.W)
nn.params.L2.W .= randn(W_shape) #initialize params#
nn.params.L2.W[1] = abs(nn.params.L2.W[1]) 
nn.params.L2.W[3] = abs(nn.params.L2.W[3]) 
nn.params.L2.W[6] = abs(nn.params.L2.W[6])
nn.params.L2.W[9] = -abs(nn.params.L2.W[9])
nn.params.L2.W[10] = abs(nn.params.L2.W[10])
nn.params.L2.W[11] = -abs(nn.params.L2.W[11])

# learning setting #
# const batch_size = 50
const n_epochs = 2000
optd = GeometricMachineLearning.Optimizer(AdamOptimizerWithDecay(n_epochs, 1e-2, 5e-5), nn)
opt1 = GeometricMachineLearning.Optimizer(AdamOptimizer(1e-2), nn)
opt2 = GeometricMachineLearning.Optimizer(AdamOptimizer(2e-3), nn)
opt3 = GeometricMachineLearning.Optimizer(AdamOptimizer(1e-4), nn)
λ = GlobalSection(nn.params)

initial_params_vec = vec(nn.params.L2.W) # "vec" tranforms matrix into vector #
original_shape = size(nn.params.L2.W) # remember the size of the weights #

function full_loss(p_vec::AbstractVector, all_tra::Vector)
    total_loss = 0.0

    for traj in all_tra
        total_loss += mse_loss(p_vec, traj)
    end

    return total_loss/length(all_tra)
end

function mse_loss(p_vec, traj)
    W_matrix = reshape(p_vec, original_shape) 
    current_ps = (L1 = NamedTuple(), L2 = (W = W_matrix,))

    current_batch = traj
    z_input = current_batch[:, 2:end-1]
    dz_dt_true = (traj[:, 3:end] - traj[:, 1:end-2]) / (2 * dt)
    num_samples = size(z_input, 2)

    T = eltype(p_vec) # type of "p_vec" (Float64 or ForwardDiff.Dual) #
    dz_dt_pred = Matrix{T}(undef, N_features, num_samples)

    for j in 1:num_samples
        z_sample = z_input[:, j]
        hamiltonian_func(vec_in) = nn(reshape(vec_in, :, 1), current_ps)[1, 1]
        grad_H = ForwardDiff.gradient(hamiltonian_func, z_sample)
        
        dH_dq = grad_H[1:N_dim]
        dH_dp = grad_H[(N_dim + 1):end]
        dz_dt_pred[:, j] = vcat(dH_dp, -dH_dq)
    end

    return mean(abs2, dz_dt_pred - dz_dt_true)
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
# p_vec = [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, -2.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0]
# loss = full_loss(p_vec, input)
thre = 0.02


println("start learning")

for ep in 1:n_epochs
    p_vec[1] = 0    
    loss_grad(p_vec) = full_loss(p_vec, input)
    gs_vec = ForwardDiff.gradient(loss_grad, p_vec)
    
    # gs_vec -> NamedTuple #
    gs_matrix = reshape(gs_vec, original_shape)
    gs_namedtuple = (L1 = NamedTuple(), L2 = (W = gs_matrix,))

    if ep == 1 # revew the parameters #
        SINDyAE_optimization_step!(opt1, λ, nn.params, gs_namedtuple)
    elseif err[ep-1] > 0.001
        SINDyAE_optimization_step!(opt1, λ, nn.params, gs_namedtuple)
    elseif err[ep-1] > 0.00001
        SINDyAE_optimization_step!(optd, λ, nn.params, gs_namedtuple)
    else
        SINDyAE_optimization_step!(opt3, λ, nn.params, gs_namedtuple)
    end
    
    global p_vec = vec(nn.params.L2.W) # nn.params -> vector #
    nn.params.L2.W .-= 1e-3 * nn.params.L2.W # update the parameters
    nn.params.L2.W[abs.(nn.params.L2.W).< thre] .= 0.0

    loss = full_loss(p_vec, input)
    err[ep] = loss
    println("nn.params.L2.W=", nn.params.L2.W)
    println("err=", err[ep])
end



# display the function #

variable_names = ["q", "p"] 
learned_weights = vec(nn.params.L2.W)
exp_list = nn.model.layers[1].exponents

hamiltonian_string = "H = "
for (i, weight) in enumerate(learned_weights)
    if abs(weight) > 1e-3 # neglect small params #
        
        if weight > 0 && i > 1 # display sign #
            hamiltonian_string *= " + "
        else
            hamiltonian_string *= " "
        end
        
        hamiltonian_string *= string(round(weight, digits=4))
        
        term_string = "" # display term from exp vector #
        for (j, exponent) in enumerate(exp_list[i])
            if exponent > 0
                term_string *= " * " * variable_names[j]
                if exponent > 1
                    term_string *= "^" * string(exponent)
                end
            end
        end
        hamiltonian_string *= term_string
    end
end

println(hamiltonian_string)

