
using InteractiveUtils
using LinearAlgebra
using Plots
using ForwardDiff

const q=1.0
const m=1.0
const E=[1.0, 0.0, 0.0]
const B=[0.0, 0.0, -2.0]

function lorentz_force(v)
	return q*(E+cross(v, B))
end

function hamiltonian(x, y, z, vx, vy, vz)
    A = [2y, 0.0, 0.0]
	Φ = -x              
    v = [vx, vy, vz]
    p = m * v
    # p_qA = p .- q .* A
    kin = dot(p, p) / (2m)
    pot = q * Φ
	# pot = 0
    return kin + pot
end

function hamiltonian_euler(r, p) #
    A = [2*r[2], 0.0, 0.0]
	Φ = -r[1]              
    p_qA = p .- q .* A
    kin = dot(p_qA, p_qA) / (2m)
    pot = q * Φ
    return kin + pot
end

# ╔═╡ 5b34a442-4429-4356-8f3f-9fa432920b44
∂H_∂r(r, p) = ForwardDiff.gradient(r -> hamiltonian_euler(r,p), r)

# ╔═╡ 163a837f-9211-4f8d-abd7-23b541e5dc3f
∂H_∂p(r, p) = ForwardDiff.gradient(p -> hamiltonian_euler(r,p), p)

# ╔═╡ 680d5b5d-b900-4c2f-aa2f-7e1c44e658a8
function leapfrog_simulation(r0, v0, dt, N)
    r = zeros(3, N+1)
    v = zeros(3, N+1)
	p = zeros(3, N+1)
	H = zeros(N+1)
	v_half = zeros(3)
	
    r[:,1] = r0
    v[:,1] = v0
	p[:,1] = m * v[:,1] + q * [2*r[2,1],0,0]
	r[:,2] = r0 + 2 * dt * v0
	v[:,2] = v0 + 2 * dt * lorentz_force(v0) / m
	p[:,2] = m * v[:,2] + q * [2*r[2,2],0,0]
    H[1] = hamiltonian(r[1,1], r[2,1], r[3,1], v[1,1], v[2,1], v[3,1])
	H[2] = hamiltonian(r[1,2], r[2,2], r[3,2], v[1,2], v[2,2], v[3,2])

    for i in 1:N-1
		v[:,i+2] = v[:,i] + 2 * dt * lorentz_force(v[:,i+1]) / m
		r[:,i+2] = r[:,i] + 2 * dt * v[:,i+1]
		p[:,i+2] = m * v[:,i+2] + q * [2*r[2,i+2],0,0]
 		v_half = zeros(3)

		H[i+2] = hamiltonian(r[1,i+2], r[2,i+2], r[3,i+2], v[1,i+2], v[2,i+2], v[3,i+2])
    end
	
    # v[:,end] = v_half - 0.5 * dt * lorentz_force(v_half) / m
    return r, v, H, p
end

# ╔═╡ 3b7ef14e-60ce-49b2-8387-3adb2a952e17
# ╠═╡ disabled = true
#=╠═╡
function leapfrog_simulation(r0, v0, dt, N)
    r = zeros(3, N+1)
    v = zeros(3, N+1)
	H = zeros(N+1)
	v_half = zeros(3)
	
    r[:,1] = r0
    v[:,1] = v0 + 0.5 * dt * lorentz_force(v0) / m
    H[1] = hamiltonian(r[1,1], r[2,1], r[3,1], v[1,1], v[2,1], v[3,1])

    for i in 1:N
		# v_half = v[:,i] + 0.5 * dt * lorentz_force(v[:,i])
        v[:,i+1] = v[:,i] + 0.5 * dt * lorentz_force(v_half) / m
		r[:,i+1] = r[:,i] + dt * v[:,i]
		v_half = zeros(3)

		H[i+1] = hamiltonian(r[1,i+1], r[2,i+1], r[3,i+1], v[1,i+1], v[2,i+1], v[3,i+1])
    end
	
    # v[:,end] = v_half - 0.5 * dt * lorentz_force(v_half) / m
    return r, v, H
end
  ╠═╡ =#

# ╔═╡ 7b12645b-8a5c-4b56-ba30-e6cbf1c30553
function boris_simulation(r0, v0, dt, N)
    r = zeros(3, N+1)
    v = zeros(3, N+1)
	H = zeros(N+1)
	v_nega = zeros(3)
	v_zero = zeros(3)
	v_posi = zeros(3)
	
    r[:,1] = r0
    v[:,1] = v0 + 0.5 * dt * lorentz_force(v0) / m
    H[1] = hamiltonian(r[1,1], r[2,1], r[3,1], v[1,1], v[2,1], v[3,1])

    for i in 1:N
		v_nega = v[:,i] + 0.5 * dt * q * E / m
		v_zero = v_nega + 0.5 * dt * q * cross(v_nega, B) / m
		v_posi = v_nega + 0.5 * dt * lorentz_force(v[:,i])
        v[:,i+1] = v_nega + 0.5 * dt * q * E / m
		r[:,i+1] = r[:,i] + dt * v[:,i+1]
		v_nega = zeros(3)
		v_zero = zeros(3)
		v_posi = zeros(3)

		H[i+1] = hamiltonian(r[1,i+1], r[2,i+1], r[3,i+1], v[1,i+1], v[2,i+1], v[3,i+1])
    end
	
    # v[:,end] = v_half - 0.5 * dt * lorentz_force(v_half) / m
    return r, v, H
end

# ╔═╡ 005e4e74-d17b-4b98-91d5-cdbc74be67a8
function symplectic_euler(r0, v0, dt, N)
    r = zeros(3, N+1)
	v = zeros(3, N+1)
    p = zeros(3, N+1)

	E = [1.0, 0.0, 0.0]
	B = [0.0, 0.0, -2.0]
    r[:,1] = r0
	v[:,1] = v0 - dt.* q .* (E+cross(v0, B)) ./ (2*m)
	A0 = [2 * r[2,1], 0.0, 0.0]
	p[:,1] = m .* v[:,1] .+ q .* A0

    for i in 1:N
		# p[:,i+1] = p[:,i] - dt .* ∂H_∂r(r[:,i], p[:,i])
		p[:,i+1] = p[:,i] - dt .* q .* E[:]
		# r[:,i+1] = r[:,i] + dt .* ∂H_∂p(r[:,i], p[:,i+1])
		A = [2 * r[2,i], 0.0, 0.0]
		r[:,i+1] = r[:,i] + dt .* (p[:,i+1] - q .* A[:]) ./ m
        # A = [2 * r[2,i+1], 0.0, 0.0]
		v[:,i+1] = (p[:,i+1] .- q .* A) ./ m
		A = zeros(3)
    end

    return r, v
end

# ╔═╡ 13231997-592d-45d5-92fb-170775966ecd
function midpoint_step(r, v, dt)
    a1 = lorentz_force(v) / m
	v_mid = v + 0.5 * dt * a1
	r_mid = r + 0.5 * dt * v
	a2 = lorentz_force(v_mid) / m
	new_v = v + dt * a2
	new_r = r + dt * v_mid
	return new_r, new_v
end

# ╔═╡ 5f48c85b-7618-410e-a448-ad5855559c54
function midpoint_simulation(r0, v0, dt, N)
    r = zeros(3, N+1)
    v = zeros(3, N+1)
	H = zeros(N+1)
    r[:,1] = r0
    v[:,1] = v0
    H[1] = hamiltonian(r[1,1], r[2,1], r[3,1], v[1,1], v[2,1], v[3,1])

    for i in 1:N
        r[:,i+1], v[:,i+1] = midpoint_step(r[:,i], v[:,i], dt)
		H[i+1] = hamiltonian(r[1,i+1], r[2,i+1], r[3,i+1], v[1,i+1], v[2,i+1], v[3,i+1])
	end
    return r, v, H
end

# ╔═╡ 2fc66e3d-68f3-4038-9dff-32116f294190
function save_results(filename::String, data::Matrix{Float64})
    open(filename, "w") do io
        for i in 1:size(data, 2)
            # println(io, join(data[:, i], ", "))
			println(io, join(data[:, i], " "))
        end
    end
end

# ╔═╡ 4deb63c6-ac62-48d5-bea8-21d46a6998c6
function save_resultsH(filename::String, data::Vector{Float64})
    open(filename, "w") do io
        for i in 1:size(data, 1)
            # println(io, i, join(data[i]))
			println(io, join(data[i]))
        end
    end
end

# ╔═╡ af6c63c3-963b-4a47-80a6-eaf52a39885e
function main()
    T = 20.0          # simulation time
    N = 2000
	# timpestep
    dt = T / N 
    r0 = [0.0, 0.0, 0.0]   # initial position
    v0 = [1.0, 1.0, 0.0]   # initial velocity

    # by Leapfrog method
    rl, vl, Hl, pl = leapfrog_simulation(r0, v0, dt, N)
	# rm, vm, Hm = midpoint_simulation(r0, v0, dt, N)
	# rb, vb, Hb = boris_simulation(r0, v0, dt, N)
	# rse, vsb = symplectic_euler(r0, v0, dt, N)
	
    # saving data
    save_results("exbl_x.txt", rl)
    save_results("exbl_v.txt", vl)
    # save_resultsH("exbl_H.txt", Hl)
	# save_results("exbl_p.txt", pl)
	# save_results("exbm_x.txt", rm)
    # save_results("exbm_v.txt", vm)
    # save_results("exbb_x.txt", rb)
    # save_results("exbb_v.txt", vb)

    # plot
    # p3d = plot3d(r[1,:], r[2,:], r[3,:], label="midpoint", linewidth=2, linestyle=:dash)
    # xlabel!("x"); ylabel!("y"); zlabel!("z")
    # title!("Charged Particle Trajectory in E×B Field")
	# savefig(p3d, "exb_trajectory_3d.png")
	p2dl = plot(rl[1,:], rl[2,:], label="leapfrog", linewidth=2, linestyle=:dash, fmt=:png)
    xlabel!("x"); ylabel!("y")
    title!("Charged Particle Trajectory in E×B Field")
	savefig(p2dl, "exbl_trajectory.png")
	
	# p2dm = plot(rm[1,:], rm[2,:], label="midpoint", linewidth=2, linestyle=:dash, fmt=:png)
    # xlabel!("x"); ylabel!("y")
    # title!("Charged Particle Trajectory in E×B Field")
	# savefig(p2dm, "exbm_trajectory.png")
	
	# p2db = plot(rb[1,:], rb[2,:], label="boris", linewidth=2, linestyle=:dash, fmt=:png)
    # xlabel!("x"); ylabel!("y")
    # title!("Charged Particle Trajectory in E×B Field")
	# savefig(p2dm, "exbb_trajectory.png")

	# p2dse = plot(rse[1,:], rse[2,:], label="symp_euler", linewidth=2, linestyle=:dash, fmt=:png)
    # xlabel!("x"); ylabel!("y")
    # title!("Charged Particle Trajectory in E×B Field")
	# savefig(p2dse, "exbse_trajectory.png")
	
	t = range(0, T, length=N+1)
    Hl_pic=plot(t, Hl, label="leapfrog", xlabel="Time", ylabel="H", title="Hamiltonian vs Time")
	savefig(Hl_pic, "exbl_Hamiltonian.png")    
	# Hm_pic=plot(t, Hm, label="midpoint", xlabel="Time", ylabel="H", title="Hamiltonian vs Time")
	# savefig(Hm_pic, "exbm_Hamiltonian.png")
	# Hb_pic=plot(t, Hb, label="boris", xlabel="Time", ylabel="H", # title="Hamiltonian vs Time")
	# savefig(Hb_pic, "exbb_Hamiltonian.png")

end
