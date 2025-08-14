
using GeometricMachineLearning
using GeometricMachineLearning: AbstractExplicitLayer, Architecture, NeuralNetworkBackend
using GeometricMachineLearning.AbstractNeuralNetworks: Initializer
using Random: AbstractRNG
import Combinatorics

"""
    exponent_vectors(n, d)

Compute all possible exponents of degree up to `d` of monomials with `n` variables.

Number of coefficients is `binomial(n + d, d)`.
"""
function exponent_vectors(n::Integer, d::Integer)
    exponents = []
    function backtrack(current::Vector{Int}, pos::Int, remaining::Int)
        if pos > n
            push!(exponents, copy(current))
            return nothing
        end
        for i in 0:remaining
            current[pos] = i
            backtrack(current, pos + 1, remaining - i)
        end
    end
    
    backtrack(zeros(Int, n), 1, d)
    println("exp=", exponents)
    exponents
end

struct PolynomialLayer{M, N} <: AbstractExplicitLayer{M, N}
    degree::Int
    exponents::Vector{Vector{Int}}

    function PolynomialLayer(degree::Integer, num_features::Integer)
        exps = exponent_vectors(num_features, degree)
        new{num_features, length(exps)}(degree, exps)
    end
end


function (layer::PolynomialLayer)(X_input, ps::NamedTuple)
    X = X_input

    # Zygoteが最も理解しやすい、ただの明示的なforループを使ったヘルパー関数
    function calculate_term_explicitly(x_col, e)
        # 乗算の単位元から始める
        term = 1.0
        # 単純なforループで値を一つずつ掛けていく
        for k in 1:length(x_col)
            term *= x_col[k]^e[k]
        end
        return term
    end

    # 各サンプル(列)に対して、全多項式を計算する関数
    calculate_all_exponents_for_sample(x_col) = map(e -> calculate_term_explicitly(x_col, e), layer.exponents)
    
    # 全サンプルに対して上の関数を適用し、結果を結合(hcat)して行列にする
    # この部分は変更なし
    return hcat(map(calculate_all_exponents_for_sample, eachcol(X))...)
end


GeometricMachineLearning.initialparameters(rng::AbstractRNG, initializer::Initializer, l::PolynomialLayer, ::NeuralNetworkBackend, ::Type{T}; kwargs...) where T = NamedTuple()
struct RadialLayer{M, N, VT <: Tuple} <: AbstractExplicitLayer{M, N}
    functions::VT

    RadialLayer(functions::FT) where {FT} = new{length(functions), length(functions), FT}(functions)
end


# input: (num_features, num_samples)
function (layer::RadialLayer)(input::Matrix{Float64}, ps::NamedTuple)
    num_features, num_samples = size(input)
    @assert num_features == length(layer.functions)

    # Construct result without mutation
    output = [layer.functions[i](input[i, j]) for i in 1:num_features, j in 1:num_samples]

    return output
end

GeometricMachineLearning.initialparameters(rng::AbstractRNG, initializer::Initializer, l::RadialLayer, ::NeuralNetworkBackend, ::Type{T}; kwargs...) where T = NamedTuple()
const radial_functions = (
    # x -> x,
    # x -> x^2,
    # atan,
    sin,
    cos,
    # identity,
    exp,
    x -> log(abs(x) + 1e-5),
    # x -> 1 / (1 + x^2)
)

struct PRPModel{RFT} <: Architecture
    N_features::Int
    degrees::Tuple{Int, Int}
    r_funs::RFT

    PRPModel(n_Features::Int;degrees::Tuple{Int, Int}=(2,2), r_funs::RFT = radial_functions) where {RFT} = new{typeof(r_funs)}(n_Features, degrees, r_funs)
end

function GeometricMachineLearning.Chain(arch::PRPModel)
    n_r_funs = length(arch.r_funs)
    @show n_r_funs
    NP1_output = Combinatorics.binomial(arch.N_features + arch.degrees[1], arch.degrees[1])
    NP2_output = Combinatorics.binomial(n_r_funs + arch.degrees[2], arch.degrees[2])
    @show NP1_output
    @show NP2_output

    P1_layer = PolynomialLayer(arch.degrees[1], arch.N_features)
    Plinear_layer = Linear(NP1_output, 1) # same dimension as the number of radial functions
    # R_layer = RadialLayer(arch.r_funs)
    # Rlinear_layer = Linear(n_r_funs, 1)
    # P2_layer = PolynomialLayer(arch.degrees[2], n_r_funs)
    # P2linear_layer = Linear(NP2_output, 1)

    # Chain(P1_layer, Plinear_layer, R_layer, P2_layer, P2linear_layer)
    Chain(P1_layer, Plinear_layer)
end

