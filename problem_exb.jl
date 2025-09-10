module exb

    using EulerLagrange
    using GeometricEquations
    using GeometricSolutions
    using Parameters
    
    const tspan = (0.0, 10.0) # 20
    const tstep = 0.01 # 0.05

    const default_parameters = (Q=1.0, m=1.0)

    # const q₀ = [2., 0., 0.]
    # const p₀ = [-1., 0., 0.]    
    
    function hamiltonian(::Number, q::AbstractArray, p::AbstractArray, params)
        Q = params.Q; m = params.m
        Ax = 2*q[2]; Ay = 0.0; Az = 0.0 # A = (2y,0,0)
        Πx = p[1] - Q*Ax
        Πy = p[2] - Q*Ay
        Πz = p[3] - Q*Az
        ϕ = -q[1] # ϕ = (-x,0,0)
        
        return (Πx^2 + Πy^2 + Πz^2)/(2m) + Q*ϕ
    end

    function hamiltonian_system(parameters::NamedTuple)
        t, q, p = hamiltonian_variables(3)
        sparams = symbolize(parameters)
        HamiltonianSystem(hamiltonian(t, q, p, sparams), t, q, p, sparams)
    end

    _parameters(p::NamedTuple) = p
    _parameters(p::AbstractVector) = p[begin]

    function hodeensemble(q₀ = q₀, p₀ = p₀; tspan=tspan, tstep=tstep, parameters=default_parameters)
        eqs = functions(hamiltonian_system(_parameters(parameters)))
        HODEEnsemble(eqs.v, eqs.f, eqs.H, tspan, tstep, q₀, p₀; parameters=parameters)
    end

end
