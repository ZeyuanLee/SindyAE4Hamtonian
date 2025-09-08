# using only forwarddiff, for harmonic oscilator, for multi data #
# data from implicit mid point in program file #
# with Autoencoder, the input data includes qz and pz #

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
files = ["./problem/exb61.txt", "./problem/exb62.txt", "./problem/exb63.txt", "./problem/exb64.txt"]
input = [collect(transpose(readdlm(file))) for file in files]

# setting model #
N_samples = size(input[1],2)
N_features = size(input[1],1) # for ex 
l_features = 4
output = zeros(N_features, N_samples) #??
N_dim = N_features ÷ 2
l_dim = l_features ÷ 2

struct ENSD{E, H, D}
    encoder::E
    hamiltonian_model::H
    decoder::D
end
Functors.@functor ENSD ##

ec = NeuralNetwork(Chain(Dense(N_features,10,tanh), Dense(10,10,tanh), Dense(10,l_features,tanh)))
ns = NeuralNetwork(Chain(Chain(PRPModel(l_features)).layers...,)) # NestedSindy
dc = NeuralNetwork(Chain(Dense(l_features,N_features,tanh)))

sae = ENSD(ec, ns, dc)

# W_shape = size(sae.ns.params.L2.W)
# sae.ns.params.L2.W .= randn(W_shape) #initialize params#

# learning setting #
# const batch_size = 50
const n_epochs = 2000
optd = GeometricMachineLearning.Optimizer(AdamOptimizerWithDecay(n_epochs, 1e-2, 5e-5), sae)
λ = GlobalSection(GeometricMachineLearning.parameters(sae))

ini_p_vec = vec() # "vec" tranforms matrix into vector #
orig_shape = size(sae.ns.params.L2.W) # remember the size of the weights #

function full_loss(p_vec::AbstractVector, all_tra::Vector, sae::ENSD)
    total_loss = 0.0

    for traj in all_tra
        total_loss += mse_loss(p_vec, traj, sae)
    end

    return total_loss/length(all_tra)
end

function mse_loss(p_vec, traj, sae)
    # W_matrix = reshape(p_vec, original_shape) # not needed ? #
    # current_ps = (L1 = NamedTuple(), L2 = (W = W_matrix,))

    current_batch = traj
    z_input = current_batch[:, 2:end-1]
    dz_dt_true = (traj[:, 3:end] - traj[:, 1:end-2]) / (2 * dt)
    num_samples = size(z_input, 2)

    T = eltype(p_vec) # type of "p_vec" (Float64 or ForwardDiff.Dual) #
    dz_dt_pred_l = Matrix{T}(undef, l_features, num_samples)
    dz_dt_pred_o = Matrix{T}(undef, N_features, num_samples)

    for j in 1:num_samples
        z_sample = z_input[:, j] #6D

        l_input = sae.encoder(z_sample, sae.encoder.params)[:,1]

        hamiltonian_func_latent(l_vec_in) = sae.ns(reshape(l_vec_in, :, 1), sae.hamiltonian_model.params)[1, 1]
        grad_H_l = ForwardDiff.gradient(hamiltonian_func_latent, l_input)
        
        dH_dq_l = grad_H_l[1:l_dim]
        dH_dp_l = grad_H_l[(l_dim + 1):end]
        dz_dt_pred_l[:, j] = vcat(dH_dp_l, -dH_dq_l)

        # decoder_input = dz_dt_pred_l[:, j]
        dz_dt_pred_o[:, j] = sae.decoder(dz_dt_pred_l, ae.decoder.params)[:,1]
    end

    return mean(abs2, dz_dt_pred_o - dz_dt_true)
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
p_vec = ini_p_vec
# p_vec = [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, -2.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0]
# loss = full_loss(p_vec, input)
thre = 0.02


println("start learning")

for ep in 1:n_epochs
    p_vec[1] = 0    
    loss_grad(p_vec) = full_loss(p_vec, input, sae)
    gs_vec = ForwardDiff.gradient(loss_grad, p_vec)
    
    # gs_vec -> NamedTuple #
    gs_matrix = reshape(gs_vec, original_shape)
    gs_namedtuple = (L1 = NamedTuple(), L2 = (W = gs_matrix,))

    SINDyAE_optimization_step!(optd, λ, nn.params, gs_namedtuple)
    
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
