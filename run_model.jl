using Pkg
Pkg.activate(".")
include("model.jl")

using GeometricMachineLearning:_optimization_step!,NeuralNetworkParameters

N_samples = 5
N_features = 2
arch = PRPModel(N_features)
nn = NeuralNetwork(Chain(Dense(2,10,tanh),Dense(10,10,tanh),Dense(10,2,tanh),# encoder 
                    Chain(arch).layers..., # Nested Sindy 
                    Dense(1,10,tanh),Dense(10,2,identity))) # decoder


const batch_size = 10
const n_epochs = 50
opt = GeometricMachineLearning.Optimizer(AdamOptimizerWithDecay(n_epochs, 1e-3, 5e-5), nn)

# loss function, could be changed into loss function we discussed
function mse_loss(x,y::AbstractArray{T},NN,ps) where T
    y_pred = NN(x,ps)
    mse_loss = mean(abs,y_pred - y)
    return mse_loss
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

# Some random data for testing
input = rand(2, batch_size)
output = rand(2, batch_size)

using Zygote
using Statistics
err = zeros(50)
λ = GlobalSection(nn.params)


for ep in 1:n_epochs
    gs = Zygote.gradient(p -> mse_loss(input, input, nn, p)[1], nn.params)[1]
    SINDyAE_optimization_step!(opt,λ, nn.params, gs)
    err[ep] = mse_loss(input, input, nn, nn.params)[1]
end

