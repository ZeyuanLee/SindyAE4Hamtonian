### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils
using MLJ
using DelimitedFiles
using SymbolicRegression
using SymbolicRegression: ValidVector
using MLJBase: machine, fit!, predict, report
using SymbolicUtils
using Plots


# for i in 1:4 
    filename = "exbl_x.txt"
    qm = readdlm(filename)
    # append!(q, [Tuple(qm[j, :]) for j in 1:size(qm, 1)])
    q = [Tuple(qm[j, :]) for j in 1:size(qm, 1)]
# end

size(q)


N = size(q,1)

# for i in 1:4  # ← 読み込むファイル数に応じて N を指定
    filename = "exbl_v.txt"
    pm = readdlm(filename)
    # append!(p, [Tuple(pm[j, :]) for j in 1:size(pm, 1)])
    p = [Tuple(pm[j, :]) for j in 1:size(pm, 1)]
# end

qdotm = zeros(3,N)


for i in 1:N-1
    q1 = collect(q[i])      # Tuple → Vector
    q2 = collect(q[i+1])    # Tuple → Vector
    qdotm[:, i] = q2 .- q1  # 要素ごとの差分
end


qdot = [Tuple(qdotm[:,i]) for i in 1:N]

pdotm = zeros(3,N)


for i in 1:N-1
    p1 = collect(p[i])      # Tuple → Vector
    p2 = collect(p[i+1])    # Tuple → Vector
    pdotm[:, i] = p2 .- p1  # 要素ごとの差分
end

pdot = [Tuple(pdotm[:,i]) for i in 1:N]

co = [(q[1], q[2], p[1], p[2]) for (q,p) in zip(qdot, pdot)]
data = (; q, p, qdot, pdot, co)

X = (;
    q_x=[qi[1] for qi in data.q],
    q_y=[qi[2] for qi in data.q],
    # q_z=[qi[3] for qi in data.q],
    p_x=[pi[1] for pi in data.p],
    p_y=[pi[2] for pi in data.p],
    # p_z=[pi[3] for pi in data.p],
    qdot_x=[qdoti[1] for qdoti in data.qdot],
    qdot_y=[qdoti[2] for qdoti in data.qdot],
    # qdot_z=[qdoti[3] for qdoti in data.qdot],
    pdot_x=[pdoti[1] for pdoti in data.pdot],
    pdot_y=[pdoti[2] for pdoti in data.pdot]
    # pdot_z=[pdoti[3] for pdoti in data.pdot]
)

keys(X)

struct DC{T}
	qdot_x::T
	qdot_y::T
	# qdot_z::T
	pdot_x::T
	pdot_y::T
	# pdot_z::T
end

y = [DC(co...) for co in data.co]

variable_names = ["q_x", "q_y", "q_z", "p_x", "p_y", "p_z"]

function diff_Hami((;H), (q_x, q_y, p_x, p_y, qdot_x, qdot_y, pdot_x, pdot_y))
#function diff_Hami((;H), (q_x, q_y, q_z, p_x, p_y, p_z, qdot_x, qdot_y, qdot_z, pdot_x, pdot_y, pdot_z))
    # ;Hにする必要あり
    # designate dt
	dt = 0.01

	# _H = H(q_x, q_y, q_z, p_x, p_y, p_z)
    _H = H(q_x, q_y, p_x, p_y)
	
	#q_x_1 の取り出し temp コードのサンプル依存性はどこに？
	# H0 = H(q_x, q_y, q_z, p_x, p_y, p_z)
	H0 = H(q_x, q_y, p_x, p_y)
	
    # 2. 有限差分で ∂H/∂p と ∂H/∂q を計算
    # dpxH = (H(q_x, q_y, q_z, p_x + pdot_x, p_y, p_z) - H0) / dt
	# .+ を用いると、lenghを使うためエラーが起こる
	# dpyH = (H(q_x, q_y, q_z, p_x, p_y + pdot_y, p_z) - H0) / dt
	# dpzH = (H(q_x, q_y, q_z, p_x, p_y, p_z + pdot_z) - H0) / dt
	dpxH = (H(q_x, q_y, p_x + pdot_x, p_y) - H0) / dt
	dpyH = (H(q_x, q_y, p_x, p_y + pdot_y) - H0) / dt
	# dpzH = (H(q_x[i], q_y[i], q_z[i], p_x[i], p_y[i], p_z[i+1]) - H(q_x[i],        q_y[i], q_z[i], p_x[i], p_y[i], p_z[i])) / dt
    # dqxH = -(H(q_x + qdot_x, q_y, q_z, p_x, p_y, p_z) - H0) / dt
    # dqyH = -(H(q_x, q_y + qdot_y, q_z, p_x, p_y, p_z) - H0) / dt
    # dqzH = -(H(q_x, q_y, q_z + qdot_z, p_x, p_y, p_z) - H0) / dt
	dqxH = -(H(q_x + qdot_x, q_y, p_x, p_y) - H0) / dt
	dqyH = -(H(q_x, q_y + qdot_y, p_x, p_y) - H0) / dt

    # 4. 結果を ValidVector に包んで返す?
	
	#dc = [DC(dpxH, dpyH, dqxH, dqyH) for (dpxH, dpyH, dqxH, dqyH) in zip(dpxH.x, dpyH.x, dqxH.x, dqyH.x)]
	dc = [DC((dqxH, dqyH, dpxH, dpyH)...) for (dpxH, dpyH, dqxH, dqyH) in zip(dpxH.x, dpyH.x, dqxH.x, dqyH.x)]

	
    # return (dpxH, dpyH, dpzH, dqxH, dqyH, dqzH)
	# return (dpxH, dpyH, dqxH, dqyH)
	# return ValidVector(DC(dpxH, dpyH, dqxH, dqyH), isvalid)
	ValidVector(dc, dpxH.valid && dpyH.valid && dqxH.valid && dqyH.valid)
	# ValidVector(DC, dpxH.valid && dpyH.valid && dpzH.valid && dqxH.valid && dqyH.valid && dqzH.valid)
    # ValidVector(dpxH.valid && dpyH.valid && dpzH.valid && dqxH.valid && dqyH.valid && dqzH.valid)
	
end

# ╔═╡ 523ece49-f4d8-4494-bb69-df0a164616c2
structure = TemplateStructure{(:H,)}(diff_Hami)

# ╔═╡ 632ca2fd-2721-44cb-a4dc-461ee26fe737
# H"," が重要らしい

# ╔═╡ 4a965f6e-71d2-46f3-b6fe-966f14a9b42b
# ハミルトニアンの表式を探す

# ╔═╡ 904862d5-df89-460e-b914-4c497610536f
# ╠═╡ disabled = true
#=╠═╡
f(qdot,pdot,dpH,dqH) = (qdot-dpH)^2 + (pdot-dqH)^2
  ╠═╡ =#

# ╔═╡ e200c9ce-5be2-4040-91b4-8c789f5ab5ae
# ╠═╡ disabled = true
#=╠═╡
options = Options(
    binary_operators=(+, -, *, /),
    unary_operators=(sin, cos, exp, sqrt),
    # niterations=500,
    maxsize=30,
    batching=true,
    batch_size=30,
    expression_spec=TemplateExpressionSpec(; structure),
    loss_function_expression = my_loss_func,
)
  ╠═╡ =#

# ╔═╡ 585ef2e1-95f8-4466-ab6d-ebb63ef71467
# ╠═╡ disabled = true
#=╠═╡
function my_loss_func(expr, dataset, options)
    # expr::TemplateExpression
    preds, valid = expr(dataset.X, options)
    if !valid
        return Inf
    end
    # dataset.y: 各時刻の (dq/dt, dp/dt)
    # preds: 同じ次元数の予測
    return sum(norm.(preds .- dataset.y).^2) / size(dataset.y, 1)
end
  ╠═╡ =#

# ╔═╡ dbb8ab8f-2bee-4574-922f-f503f0e89c38
model = SRRegressor(;
	binary_operators=(+, -, *, /),
	unary_operators=(sin, cos, sqrt, exp),
	niterations=500,
	maxsize=35,
	expression_spec=TemplateExpressionSpec(; structure),
	# Note that the elmentwise loss needs to operate direcly on each row of 'y':
	#elementwise_loss=(F1, F2) -> (F1.x - F2.x)^2 + (F1.y - F2.y)^2 + (F1.z - F2.z)^2,
    elementwise_loss = (DC1, DC2) ->
    (DC2.qdot_x - DC1.qdot_x)^2 + (DC2.qdot_y - DC1.qdot_y)^2 +
    (DC2.pdot_x - DC1.pdot_x)^2 + (DC2.pdot_y - DC1.pdot_y)^2,
    # elementwise, target -> predicted 
    # loss_function_expression = my_loss_func,
	batching=true,
	batch_size=30,
	# scitype_check_level=0 なんかエラー出る
)

# ╔═╡ d5c7fe0a-0a36-455d-9065-47383c462e5a
mach=machine(model,X,y)

# ╔═╡ 48587bce-46ef-476e-b498-df4c51c9d9df
fit!(mach)

# ╔═╡ cce4787d-dd08-4841-8a9c-ff1585c3fb9b
r = report(mach)

# ╔═╡ 997114c1-99c1-44e8-a88c-54f41ef8d40b
r.equations[r.best_idx]

# ╔═╡ c9ba06b3-57ec-47a7-9ea3-28023bd17242
ypred = predict(mach, X)

# ╔═╡ 68f1c112-13aa-42b6-9c44-80a5639efe36
scatter(
    [yt.qdot_x for yt in y],         # 真の値
    [yp.qdot_x for yp in ypred],   # 予測値（.x は ValidVector の中身）
    xlabel = "True q̇ₓ",
    ylabel = "Predicted q̇ₓ",
    label = "q̇ₓ",
    legend = :topleft
)


q = []
p = []
