# using ForwardDiff
include("problem_exb.jl")    
using GeometricIntegrators: ImplicitMidpoint, integrate
using GeometricSolutions
using Plots
using DelimitedFiles

q₀_vec = [[0., 0., 0.],
          [1., 0., 0.],
          [2., 0., 0.],
          [3., 0., 0.]]
p₀_vec = [[2., 0., 0.],
          [-2., 0., 0.],
          [1., 0., 0.],
          [-1., 0., 0.]]

heb = exb.hodeensemble(q₀_vec, p₀_vec) 
sol = integrate(heb, ImplicitMidpoint())

plt = plot(xlabel="x", ylabel="y", legend=false)
for i in 1:4
    s = sol[i]
    x = getindex.(s.q, 1)  
    y = getindex.(s.q, 2) 
    plot!(plt, x, y)
    Q = reduce(hcat, (getindex.(s.q, j) for j in 1:3))
    P = reduce(hcat, (getindex.(s.p, j) for j in 1:3))
    data = hcat(Q, P)
    writedlm("exb6$(i).txt", data, ' ')
end
display(plt)