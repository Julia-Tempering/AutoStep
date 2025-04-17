using Random
using Distributions

function fMALA(x, z, θ, target, auxtarget)
	zh = z + θ/2*gradlogpdf(target, x)
	xp = x + θ*zh
	zp = -(zh + θ/2*gradlogpdf(target, xp))
	return xp, zp
end

function fRWMH(x, z, θ, target, auxtarget)
	return x+θ*z, -z
end

function μ(x, z, a, b, θ0, f, target, auxtarget, symmetric)
    xp, zp = f(x, z, θ0, target, auxtarget)
	ℓ = logpdf(target, xp) + logpdf(auxtarget, zp) - logpdf(target, x) - logpdf(auxtarget, z)
	cost = 1
	v = symmetric ? Int(abs(ℓ) < abs(log(b))) - Int(abs(ℓ) > abs(log(a))) : Int(ℓ > log(b)) - Int(ℓ < log(a))
	if v == 0
		return 0, cost
	end
	j = 0
	while true
		j += v
        xp, zp = f(x, z, θ0*(2.)^j, target, auxtarget)
	    ℓ = logpdf(target, xp) + logpdf(auxtarget, zp) - logpdf(target, x) - logpdf(auxtarget, z)
	    cost += 1
		if v > 0 && (symmetric ? (abs(ℓ) ≥ abs(log(b))) : (ℓ ≤ log(b)))
			return j-1, cost
		elseif v < 0 && (symmetric ? (abs(ℓ) ≤ abs(log(a))) : (ℓ ≥ log(a)))
			return j, cost
		end
	end
end

function η(x, z, a, b, θ0, f, target, auxtarget,symmetric)
	δ, cost = μ(x, z, a, b, θ0, f, target, auxtarget,symmetric)
	return Dirac(θ0*(2.)^δ), cost
end

function auto_step(x, f, θ0, target, auxtarget, symmetric=true)
	a0, b0 = rand(), rand()
	a = min(a0,b0)
	b = max(a0,b0)
	z = rand(auxtarget)
	ηdist, cost1 = η(x,z,a,b,θ0,f,target,auxtarget,symmetric)
	θ = rand(ηdist)
	xp, zp = f(x, z, θ, target, auxtarget)
	ηpdist, cost2 = η(xp,zp,a,b,θ0,f,target,auxtarget,symmetric)
	energyjump = logpdf(target, xp) + logpdf(auxtarget, zp) - logpdf(target, x) - logpdf(auxtarget, z)
	ℓ = energyjump + logpdf(ηpdist, θ) - logpdf(ηdist, θ)
	cost = 1 + cost1 + cost2
	if log(rand()) ≤ ℓ 
		return xp, min(0, ℓ), energyjump, cost, θ, false
	else
		return x, min(0, ℓ), 0.0, cost, θ, false
	end
end

function ηjitter(x, z, a, b, θ0, f, target, auxtarget,symmetric)
	δ, cost = μ(x, z, a, b, θ0, f, target, auxtarget,symmetric)
	xp, zp = f(x,z,θ0*(2.)^δ, target, auxtarget)
	δp, costp = μ(xp, zp, a, b, θ0, f, target, auxtarget,symmetric)
	if δp == δ
		return Dirac(θ0*(2.)^δ), cost+costp
	else
		return DiscreteNonParametric(θ0*(2.).^(min(δ,δp):max(δ,δp)), ones(abs(δp-δ)+1)/(abs(δp-δ)+1)), cost+costp
	end
end

function auto_step_jitter(x, f, θ0, target, auxtarget, symmetric=true)
	a0, b0 = rand(), rand()
	a = min(a0,b0)
	b = max(a0,b0)
	z = rand(auxtarget)
	ηdist, cost1 = ηjitter(x,z,a,b,θ0,f,target,auxtarget,symmetric)
	θ = rand(ηdist)
	xp, zp = f(x, z, θ, target, auxtarget)
    energyjump = logpdf(target, xp) + logpdf(auxtarget, zp) - logpdf(target, x) - logpdf(auxtarget, z) 
	cost = 1 + cost1
	# if the forward/backward step size are not the same, compute the reverse eta
	jittered = false
	if !(typeof(ηdist) <: Dirac)
		jittered = true
		ηpdist, cost2 = ηjitter(xp,zp,a,b,θ0,f,target,auxtarget,symmetric)
		ℓ = energyjump + logpdf(ηpdist, θ) - logpdf(ηdist, θ)
		cost += cost2
	else
		ℓ = energyjump
	end
	if log(rand()) ≤ ℓ 
		return xp, min(0, ℓ), energyjump, cost, θ, jittered
	else
		return x, min(0, ℓ), 0.0, cost, θ, jittered
	end
end


function fix_step(x, f, θ0, target, auxtarget, symmetric)
	z = rand(auxtarget)
	xp, zp = f(x, z, θ0, target, auxtarget)
	ℓ = logpdf(target, xp) + logpdf(auxtarget, zp) - logpdf(target, x) - logpdf(auxtarget, z)
	cost = 1
	if log(rand()) ≤ ℓ
		return xp, min(0, ℓ), ℓ, cost, θ0, false
	else
		return x, min(0, ℓ), 0.0, cost, θ0, false
	end
end

function mcmc(x0, kernel, f, θ0, target, auxtarget, niter, symmetric=true)
	x = copy(x0)
	xs = [x]
	cs = [0]
	logas = []
	ejumps = []
	jitters = []
	thetas = []

	for i=1:niter
		xp, logacc, ejump, cost, θ, jittered = kernel(x, f, θ0, target, auxtarget, symmetric)
		push!(xs, copy(xp))
		push!(cs, cs[i]+cost)
		push!(logas, logacc)
		push!(ejumps, ejump)
		push!(jitters, jittered)
		push!(thetas, θ)
		x = copy(xp)
	end
	return xs, cs, logas, ejumps, jitters, thetas
end
