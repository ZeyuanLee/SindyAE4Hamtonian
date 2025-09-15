# using only forwarddiff, for harmonic oscilator, for multi data #
# data from implicit mid point in program file #
# with Autoencoder, the input data includes qz and pz #
# optimized for L1-7 = all parameters #

using Pkg
Pkg.activate(".")
include("chain_diff.jl")

using GeometricMachineLearning:_optimization_step!,NeuralNetworkParameters
using AbstractNeuralNetworks
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
# sindy_arch = PRPModel(l_features) # no need to add PRPModel for only polynomial layer #

# nn = NeuralNetwork(Chain(Dense(N_features,10,tanh), Dense(10,10,tanh), Dense(10,l_features,tanh),
#                    DiffHamiltonianLayer(l_features; degree=2), # NestedSindy
#                    Dense(l_features,10,tanh),Dense(10,10,tanh),Dense(10,N_features,identity)))
nn = NeuralNetwork(Chain(Dense(N_features,10,identity), Dense(10,10,identity), Dense(10,l_features,identity),
                   DiffHamiltonianLayer(l_features; degree=2), # NestedSindy
                   Dense(l_features,10,identity),Dense(10,10,identity),Dense(10,N_features,identity)))

# W_shape = size(nn.params.L4.W) # do not delete, important #
# nn.params.L4.W .= randn(W_shape) #initialize params#

params_shapes(ps) = (
    L1=(W=size(ps.L1.W), b=size(ps.L1.b)),
    L2=(W=size(ps.L2.W), b=size(ps.L2.b)),
    L3=(W=size(ps.L3.W), b=size(ps.L3.b)),
    L4=(W=size(ps.L4.W),),                    # Diff は W だけ
    L5=(W=size(ps.L5.W), b=size(ps.L5.b)),
    L6=(W=size(ps.L6.W), b=size(ps.L6.b)),
    L7=(W=size(ps.L7.W), b=size(ps.L7.b)),
)

function pack_params(ps)::Vector{Float64}
    v = Float64[]
    append!(v, vec(ps.L1.W)); append!(v, ps.L1.b)
    append!(v, vec(ps.L2.W)); append!(v, ps.L2.b)
    append!(v, vec(ps.L3.W)); append!(v, ps.L3.b)
    append!(v, vec(ps.L4.W))
    append!(v, vec(ps.L5.W)); append!(v, ps.L5.b)
    append!(v, vec(ps.L6.W)); append!(v, ps.L6.b)
    append!(v, vec(ps.L7.W)); append!(v, ps.L7.b)
    return v
end

function vec_to_params(v::AbstractVector{T}, shp) where {T}
    i = 1
    take(sz) = begin
        n   = prod(sz)
        seg = view(v, i:i+n-1)  
        i  += n
        reshape(copy(seg), sz)  # return Array{T}, reshape to sz #
    end
    L1 = (W = take(shp.L1.W), b = take(shp.L1.b))
    L2 = (W = take(shp.L2.W), b = take(shp.L2.b))
    L3 = (W = take(shp.L3.W), b = take(shp.L3.b))
    L4 = (W = take(shp.L4.W),)
    L5 = (W = take(shp.L5.W), b = take(shp.L5.b))
    L6 = (W = take(shp.L6.W), b = take(shp.L6.b))
    L7 = (W = take(shp.L7.W), b = take(shp.L7.b))
    return (L1=L1, L2=L2, L3=L3, L4=L4, L5=L5, L6=L6, L7=L7)
end

tem_ps = (nn.params.L1,nn.params.L2,nn.params.L3,nn.params.L4,nn.params.L5,nn.params.L6,nn.params.L7)
AbstractNeuralNetworks.Chain(nn.model.layers[1:7]...)(rand(6),tem_ps) ##

# learning setting #
# const batch_size = 50
const n_epochs = 500
optd = GeometricMachineLearning.Optimizer(AdamOptimizerWithDecay(n_epochs, 1e-3, 5e-5), nn)
λ = GlobalSection(nn.params)

ini_p_vec = vec(nn.params.L4.W) # "vec" tranforms matrix into vector #
orig_shape = size(nn.params.L4.W) # remember the size of the weights #

# split nn #
function split_subnets(nn)
    ch = getfield(nn, 2)                 # Chain
    ls = getfield(ch, :layers)           # (Dense, Dense, Dense, Poly, Linear, Dense, Dense, Dense)
    AE = NeuralNetwork(Chain(ls[1], ls[2], ls[3]))  # 1..3
    PD = NeuralNetwork(Chain(ls[4]))         # 4..5 (Poly -> Linear)
    DE = NeuralNetwork(Chain(ls[5], ls[6], ls[7]))  # 6..8
    return AE, PD, DE
end

function full_loss(ps, all_tra::Vector, nn)
    total_loss = 0.0

    for traj in all_tra
        total_loss += mse_loss(ps, traj, nn)
    end

    return total_loss/length(all_tra)
end

function mse_loss(ps, traj, nn; λ_rec=λ_rec)
    AE, PD, DE = split_subnets(nn)
    # total = zero(Float64)
    # count = 0

    zdot_true = (traj[:, 3:end] - traj[:, 1:end-2]) / (2 * dt)
    z_input = traj[:, 2:end-1]

    # z -> l #
    l_traj = AE(traj, (L1=ps.L1, L2=ps.L2, L3=ps.L3))
    l_input = l_traj[:, 2:end-1]
    ldot_true = (l_traj[:, 3:end] .- l_traj[:, 1:end-2]) ./ (2*dt)
    nums = size(l_input, 2)

    # calculate v_latent #
    T = eltype(ps.L4.W)
    ldot_pred = Matrix{T}(undef, l_features, nums)
    @inbounds for j in 1:nums
        lv = view(l_input, :, j)
        ldot_pred[:, j] = PD(lv, (L1=ps.L4,))[:, 1]
    end

    # corresponds to DE_output #
    zdot_pred = DE(ldot_pred, (L1=ps.L5, L2=ps.L6, L3=ps.L7))

    # re-construction error #
    z_output = DE(l_input, (L1=ps.L5, L2=ps.L6, L3=ps.L7))

    return mean(abs2, ldot_pred .- ldot_true) + λ_rec * mean(abs2, zdot_pred .- zdot_true) + mean(abs2, z_output - z_input)
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
# p_vec = ini_p_vec
shp   = params_shapes(nn.params)
p_vec = pack_params(nn.params)
λ_rec = 0.1
# p_vec = [0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, -2.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0]
thre = 0.02

println("start learning")

for ep in 1:n_epochs
    loss_vec(v) =  full_loss(vec_to_params(v, shp), input, nn)
    g = ForwardDiff.gradient(loss_vec, p_vec)

    # grad vector -> NamedTuple #
    dx = vec_to_params(g, shp)

    # renew #
    SINDyAE_optimization_step!(optd, λ, nn.params, dx)

    # thre #
    # nn.params.L4.W .-= 1e-3 .* nn.params.L4.W
    # nn.params.L4.W[abs.(nn.params.L4.W) .< thre] .= 0.0

    # substitute new params #
    p_vec = pack_params(nn.params)

    # ログ
    loss_now = full_loss(nn.params, input, nn)
    err[ep] = loss_now
    @info "ep=$ep  loss=$(loss_now)"
end



# display the function #
nn.params.L4.W
