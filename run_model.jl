using Pkg
using DelimitedFiles
using AbstractNeuralNetworks
using Zygote
# Pkg.activate(".")
include("model.jl")

files = ["orbit1.txt", "orbit2.txt", "orbit3.txt", "orbit4.txt"]
input = [collect(transpose(readdlm(file))) for file in files]

using GeometricMachineLearning:_optimization_step!,NeuralNetworkParameters

N_samples = 5
N_features = 2
arch = PRPModel(N_features)
nn = NeuralNetwork(Chain(Dense(4,10,tanh),Dense(10,10,tanh),Dense(10,2,tanh),# encoder 
Chain(arch).layers..., # Nested Sindy 
Dense(1,10,tanh),Dense(10,10,tanh),Dense(10,4,identity))) # decoder


Enc_ps = (L1 = nn.params.L1, L2 = nn.params.L2, L3 = nn.params.L3)
Enc(x) = AbstractNeuralNetworks.Chain(nn.model.layers[1:3]...)(x, Enc_ps)
Enc(input[1])


Sindy_Layer_ps = (L4 = nn.params.L4, L5 = nn.params.L5, L6 = nn.params.L6,L7 = nn.params.L7, L8 = nn.params.L8)
Sindy_Layer(x) = AbstractNeuralNetworks.Chain(nn.model.layers[4:8]...)(x, Sindy_Layer_ps)
Sindy_Layer(rand(2,5))

Dec_ps = (L9 = nn.params.L9, L10 = nn.params.L10)
Dec(x) = AbstractNeuralNetworks.Chain(nn.model.layers[9:10]...)(x, Dec_ps)
Dec(rand(1,5))  

latent_space_qp_finite_difference = (Enc(input[1][:,2]) - Enc(input[1][:,1]))/0.01
latent_space_qp_midpoint = [Enc((input[1][:,i] + input[1][:,i-1])/2) for i in 2:size(input[1],2)]
latent_space_qp_midpoint = hcat(latent_space_qp_midpoint...)

Sindy_Layer(latent_space_qp_midpoint)
jac = [Zygote.jacobian(x->Sindy_Layer(x), latent_space_qp_midpoint[:,i][:,:])[1] for i in 1:size(latent_space_qp_midpoint,2)]


const batch_size = 10
const n_epochs = 50
opt = GeometricMachineLearning.Optimizer(AdamOptimizerWithDecay(n_epochs, 0.1, 5e-5), nn)

# loss function, could be changed into loss function we discussed
function mse_loss(x,y::AbstractArray{T},NN,ps) where T
    y_pred = NN(x,ps)
    mse_loss = mean(abs,y_pred - y)
    return mse_loss
end

const latent_J = [0. 1; -1 0]  
# Alternatively, more compact:
const J = [0. 0 1 0; 0 0 0 1; -1 0 0 0; 0 -1 0 0]
const dt = 0.01

function SINDyAE_loss(x,y,NN,ps)
    # Autoencoder loss
    reconstruction_loss = mse_loss(x, y, NN, ps)

    latent_space_qp_finite_difference = (Enc(x[:,2]) - Enc(x[:,1]))/dt
    latent_space_qp_midpoint = Enc((x[:,i] + x[:,i-1])/2)
    H_jac = [Zygote.jacobian(x->Sindy_Layer(x), latent_space_qp_midpoint[:,i][:,:])[1] for i in 1:size(latent_space_qp_midpoint,2)]
    EN_Sindy_loss = latent_space_qp_finite_difference - latent_J * H_jac

    DE
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

