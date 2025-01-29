using AdvancedMH, Distributions, MCMCChains, LogDensityProblems, LinearAlgebra
using CSV, DataFrames, DelimitedFiles, JSON, Random
include("utils.jl")

# using NUTS in Turing.jl
function autostep2_sample_model(model, seed, n_rounds; max_samples = 2^25, kwargs...)
    # make model and data from the arguments
    my_data = stan_data(model)
    my_model = logdens_model(model, my_data)
    Random.seed!(seed)


    # IVY: you can use
	# run_sampler(x0, auto_step, f, θ0, target, sqrtdiagMhat, niter)
	# with f set to either fRWMH or fMALA
	# target is the LogDensityProblems.logdensity problem
	# sqrtdiagMhat is exactly on line 14 of the pseudocode in the paper -- (diag variance of x)^{-1}
	# niter = number of iterations

    # run until minESS threshold is reached
    n_logprob = 0
    miness = my_time = 0.0
    n_samples = 2
    initial_params = nothing
    local chain
    local samples
    for i in 1:n_rounds
        n_samples = 2^i
        my_time += @elapsed chain = sample(my_model, RobustAdaptiveMetropolis(), n_samples; 
            chain_type = Chains, initial_params = initial_params, progress = true)
        n_logprob += 2*n_samples # 2 logprob evaluations: one for the proposal, one for the current state
        initial_params = vec(Array(chain[end, :, :]))
    end
    samples = [chain[param] for param in names(chain)[1:end-1]] # discard aux vars
    samples = [vec(sample) for sample in samples] # convert to vectors
    samples = [collect(row) for row in eachrow(hcat(samples...))] # convert to format compatible with min_ess_all_methods
    miness = min_ess_all_methods(samples, model)
    # minKSess = min_KSess(samples, model)
    mean_1st_dim = mean(samples[1])
    var_1st_dim = var(samples[1])
    acceptance_prob = sum(1 for i in 2:n_samples if samples[i] != samples[i-1]; init=0)/(n_samples - 1)
    energy_jump_dist = mean(abs.(diff(chain[:lp], dims=1)))
    stats_df = DataFrame(
        explorer = "adaptive RWMH", model = model, seed = seed, 
        mean_1st_dim = mean_1st_dim, var_1st_dim = var_1st_dim, time=my_time, jitter_std = 0.0, n_logprob = n_logprob, 
        n_steps=0, #zero gradient
        miness=miness, minKSess = 0, acceptance_prob=acceptance_prob, step_size=0.0, n_rounds = n_rounds, 
        energy_jump_dist = energy_jump_dist)
    return samples, stats_df
end

function fMALA(x, z, θ, target, sqrtdiagM)
	zh = z + θ/2*LogDensityProblems.logdensity_and_gradient(target, x)[2]
	xp = x + θ*(zh ./ sqrtdiagM.^2)
	zp = -(zh + θ/2*LogDensityProblems.logdensity_and_gradient(target, xp)[2])
	return xp, zp
end

function fRWMH(x, z, θ, target, sqrtdiagM)
	return x+θ*z, -z
end

function μ(x, z, a, b, θ0, f, target, sqrtdiagM)
    xp, zp = f(x, z, θ0, target, sqrtdiagM)
	ℓ = LogDensityProblems.logdensity(target, xp) + sum(logpdf.(Normal.(0, sqrtdiagM), zp)) - LogDensityProblems.logdensity(target, x) - sum(logpdf.(Normal.(0, sqrtdiagM), z))
	cost = 1
	v = Int(abs(ℓ) < abs(log(b))) - Int(abs(ℓ) > abs(log(a))) 
	if v == 0
		return 0, cost
	end
	j = 0
	while true
		j += v
        xp, zp = f(x, z, θ0*(2.)^j, target, sqrtdiagM)
	    ℓ = LogDensityProblems.logdensity(target, xp) + sum(logpdf.(Normal.(0, sqrtdiagM), zp)) - LogDensityProblems.logdensity(target, x) - sum(logpdf.(Normal.(0, sqrtdiagM), z))
	    cost += 1
		if v > 0 && (abs(ℓ) ≥ abs(log(b)))
			return j-1, cost
		elseif v < 0 && (abs(ℓ) ≤ abs(log(a))) 
			return j, cost
		end
	end
end

function η(x, z, a, b, θ0, f, target, sqrtdiagM)
	δ, cost = μ(x, z, a, b, θ0, f, target, sqrtdiagM)
	return Dirac(θ0*(2.)^δ), cost
end

function auto_step(x, f, θ0, target, sqrtdiagMhat)
    dim = length(x)
	a0, b0 = rand(), rand()
	a = min(a0,b0)
	b = max(a0,b0)
	xi = rand() < 0.666 ? rand(0:1) : rand(Beta(1.0,1.0))
	sqrtdiagM = xi*sqrtdiagMhat .+ (1-xi)
	z = randn(dim) .* sqrtdiagM
	ηdist, cost1 = η(x,z,a,b,θ0,f,target,sqrtdiagM)
	θ = rand(ηdist)
	xp, zp = f(x, z, θ, target, sqrtdiagM)
	ηpdist, cost2 = η(xp,zp,a,b,θ0,f,target,sqrtdiagM)
	energyjump = LogDensityProblems.logdensity(target, xp) + sum(logpdf.(Normal.(0, sqrtdiagM), zp)) - LogDensityProblems.logdensity(target, x) - sum(logpdf.(Normal.(0, sqrtdiagM), z))
	ℓ = energyjump + logpdf(ηpdist, θ) - logpdf(ηdist, θ)
	cost = 1 + cost1 + cost2
	if log(rand()) ≤ ℓ 
		return xp, min(0, ℓ), energyjump, cost, θ
	else
		return x, min(0, ℓ), 0.0, cost, θ
	end
end

function fix_step(x, f, θ0, target, sqrtdiagMhat)
	z = rand(auxtarget)
	xp, zp = f(x, z, θ0, target, auxtarget)
	ℓ = LogDensityProblems.logdensity(target, xp) + sum(logpdf.(Normal.(0, sqrtdiagM), zp)) - LogDensityProblems.logdensity(target, x) - sum(logpdf.(Normal.(0, sqrtdiagM), z))
	cost = 1
	if log(rand()) ≤ ℓ
		return xp, min(0, ℓ), ℓ, cost, θ0
	else
		return x, min(0, ℓ), 0.0, cost, θ0
	end
end

function run_sampler(x0, kernel, f, θ0, target, sqrtdiagMhat, niter)
	x = copy(x0)
    xs = zeros(niter, length(x0))
	cs = zeros(niter)
	logas = zeros(niter)
	ejumps = zeros(niter)
	thetas = zeros(niter)
	for i=1:niter
		x, logacc, ejump, cost, θ = kernel(x, f, θ0, target, sqrtdiagMhat)
        xs[i, :] .= x
        cs[i] = (i == 1) ? cost : cs[i-1]+cost
        logas[i] = logacc
        ejumps[i] = ejump
        thetas[i] = θ
	end
	return xs, cs, logas, ejumps, thetas
end


#run_sampler([0., 0.], auto_step, fRWMH, 1.0, Funnel(1, 0.6), ones(2), 100)
