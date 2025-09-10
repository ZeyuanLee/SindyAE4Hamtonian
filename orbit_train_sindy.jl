using Pkg
# include("model_simplechain.jl")

using GeometricMachineLearning:_optimization_step!,NeuralNetworkParameters
using DelimitedFiles
using Statistics
using ForwardDiff
using LinearAlgebra
using Zygote

dt = 0.01
files = ["orbit1.txt", "orbit2.txt", "orbit3.txt", "orbit4.txt"]
input = [collect(transpose(readdlm(file))) for file in files]

# setting model #
N_samples = size(input[1],2)
N_features = size(input[1],1) # for ex 4
output = zeros(N_features, N_samples)
N_dim = N_features ÷ 2
arch = PRPModel(N_features)

nn = NeuralNetwork(Chain(Chain(arch).layers...,)) # only sindy layer
n_epochs = 3000
opt1 = GeometricMachineLearning.Optimizer(AdamOptimizer(1e-2), nn)
λ = GlobalSection(nn.params)

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

function sindy_loss(input,dz_dt_true,nn,params;λ = 1e-4)
    # H = nn(input,params)
    batch_size = size(input,2)
    err = 0
    params_norm = eval_params_norm(params)
    dz_dt_pred = zero(input)
    for sample_index in 1:batch_size
        grad_H = Zygote.gradient(x -> nn(x, params)[1], input[:,sample_index])[1]
        dH_dq = grad_H[1:N_dim]
        dH_dp = grad_H[(N_dim + 1):end]
        dz_dt_pred[:, sample_index] = vcat(dH_dp, -dH_dq)

        err += mean(abs2, dz_dt_pred - dz_dt_true)
    end
    return err + λ * params_norm
end

function finite_difference(input,dt)
    (input[:,2:end] - input[:,1:end-1]) / dt
end


function eval_params_norm(params)
    total_norm = 0.0
    for layer in values(params)
        if hasfield(typeof(layer), :W)
            total_norm += norm(layer.W)
        end
        if hasfield(typeof(layer), :b)
            total_norm += norm(layer.b)
        end
    end
    return total_norm
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

n_epochs = 3000
batch_size = 20
prune_threshold = 0.1
λ_lasso = 1e-4

input = input[1]
for ep in 1:n_epochs
    for i in 1:div(length(input), batch_size)
        batch_input = input[:,(i-1)*batch_size+1:i*batch_size+1]
        batch_output = finite_difference(batch_input, dt)
        batch_input = batch_input[:, 1:end-1]
        sindy_loss(batch_input, batch_output, nn, nn.params; λ=1e-4)
        # Compute gradients
        gs = Zygote.gradient(p -> sindy_loss(batch_input, batch_output, nn, p; λ=1e-4), nn.params)[1]

        # Perform optimization step
        SINDyAE_optimization_step!(opt1, λ, nn.params, gs)
    end

    for layer in values(nn.params)
        if hasfield(typeof(layer), :W)
            layer.W .-= λ_lasso * sign.(layer.W)
            layer.W[abs.(layer.W).< prune_threshold] .= 0.0
        end
        if hasfield(typeof(layer), :b)
            layer.b .-= λ_lasso * sign.(layer.b)
            layer.b[abs.(layer.b).< prune_threshold] .= 0.0
        end
    end


end
