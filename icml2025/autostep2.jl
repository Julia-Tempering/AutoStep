using Distributions, MCMCChains, LogDensityProblems, LinearAlgebra
using CSV, DataFrames, DelimitedFiles, JSON
include("utils.jl")

function autostep2_sample_model(model, seed, explorer, n_rounds, adapt_precond)
	# make model and data from the arguments
	my_data = stan_data(model)
	my_model = logdens_model(model, my_data)
	f = if startswith(explorer, "AutoStep RWMH")
		fRWMH
	elseif startswith(explorer, "AutoStep MALA")
		fMALA
	end
	rng = MersenneTwister(seed)

	n_logprob = n_grad_eval = 0
	miness = my_time = 0.0
	n_samples = 2
	sqrtdiagMhat = ones(LogDensityProblems.dimension(my_model))
	initial_params = zeros(LogDensityProblems.dimension(my_model))
	theta0 = 1.0
	local log_accept
	local samples
	local ejumps
	for i in 1:n_rounds
		n_samples = 2^i
		my_time += @elapsed samples, costs, log_accept, ejumps, thetas, grad_evals =
			run_sampler(initial_params, auto_step, f, theta0, my_model, sqrtdiagMhat, n_samples, rng)
		theta0 = median(thetas)
		if adapt_precond
			sqrtdiagMhat = vec(1.0 ./ std(samples, dims = 1))
		end
		n_logprob += costs[end]
		n_grad_eval += grad_evals[end]
		println(my_time)
		initial_params = samples[end, :]
	end
	mean_1st_dim = mean(samples[:, 1])
	var_1st_dim = var(samples[:, 1])
	samples = [samples[i, :] for i in 1:size(samples, 1)]
	miness = min_ess_all_methods(samples, model)
	# minKSess = min_KSess(samples, model)
	acceptance_prob = mean(exp.(log_accept))
	energy_jump_dist = mean(ejumps)
	stats_df = DataFrame(
		explorer = explorer, model = model, seed = seed,
		mean_1st_dim = mean_1st_dim, var_1st_dim = var_1st_dim, time = my_time, jitter_std = 0.0, n_logprob = n_logprob,
		n_steps = n_grad_eval, #zero gradient
		miness = miness, minKSess = 0.0, acceptance_prob = acceptance_prob, step_size = theta0, n_rounds = n_rounds,
		energy_jump_dist = energy_jump_dist)
	println("Done!")
	return samples, stats_df
end

function fMALA(x, z, θ, target, sqrtdiagM)
	zh = z + θ / 2 * LogDensityProblems.logdensity_and_gradient(target, x)[2]
	xp = x + θ * (zh ./ sqrtdiagM .^ 2)
	zp = -(zh + θ / 2 * LogDensityProblems.logdensity_and_gradient(target, xp)[2])
	return xp, zp, 2
end

function fRWMH(x, z, θ, target, sqrtdiagM)
	return x + θ * (z ./ sqrtdiagM .^ 2), -z, 0
end

function μ(x, z, a, b, θ0, f, target, sqrtdiagM)
	xp, zp, grad_eval = f(x, z, θ0, target, sqrtdiagM)
	ℓ = try
		LogDensityProblems.logdensity(target, xp) + sum(logpdf.(Normal.(0, sqrtdiagM), zp)) - LogDensityProblems.logdensity(target, x) - sum(logpdf.(Normal.(0, sqrtdiagM), z))
	catch e
		NaN
	end
	cost = 1
	v = Int(abs(ℓ) < abs(log(b))) - Int(abs(ℓ) > abs(log(a)))
	if v == 0 || isnan(ℓ)
		return 0, cost, grad_eval
	end
	j = 0
	while true
		j += v
		xp, zp, grad_eval_temp = f(x, z, θ0 * (2.0)^j, target, sqrtdiagM)
		grad_eval += grad_eval_temp
		ℓ = try
			LogDensityProblems.logdensity(target, xp) + sum(logpdf.(Normal.(0, sqrtdiagM), zp)) - LogDensityProblems.logdensity(target, x) - sum(logpdf.(Normal.(0, sqrtdiagM), z))
		catch e
			NaN
		end
		cost += 1
		if isnan(ℓ) || ℓ == 0.0
			return 0, cost, grad_eval
		elseif v > 0 && (abs(ℓ) ≥ abs(log(b)))
			return j - 1, cost, grad_eval
		elseif v < 0 && (abs(ℓ) ≤ abs(log(a)))
			return j, cost, grad_eval
		end
	end
end

function η(x, z, a, b, θ0, f, target, sqrtdiagM)
	δ, cost, grad_eval = μ(x, z, a, b, θ0, f, target, sqrtdiagM)
	return Dirac(θ0 * (2.0)^δ), cost, grad_eval
end

function auto_step(x, f, θ0, target, sqrtdiagMhat, rng)
	dim = length(x)
	# random mixing of preconditioner
	u = rand(rng)
	xi = u < 1//3 ? 0 : (u < 2//3 ? 1 : rand(rng))
	sqrtdiagM = (1 - xi) .* sqrtdiagMhat .+ xi
	sqrtdiagM .= ifelse.(sqrtdiagM .== 0, 1, sqrtdiagM) # if sqrtdiagM is 0, replace with 1
	a0, b0 = rand(rng), rand(rng)
	a = min(a0, b0)
	b = max(a0, b0)
	z = randn(rng, dim) .* sqrtdiagM
	ηdist, cost1, grad_eval1 = η(x, z, a, b, θ0, f, target, sqrtdiagM)
	θ = rand(rng, ηdist)
	xp, zp, grad_eval2 = f(x, z, θ, target, sqrtdiagM)
	ηpdist, cost2, grad_eval3 = η(xp, zp, a, b, θ0, f, target, sqrtdiagM)
	energyjump = try
		LogDensityProblems.logdensity(target, xp) + sum(logpdf.(Normal.(0, sqrtdiagM), zp)) - LogDensityProblems.logdensity(target, x) - sum(logpdf.(Normal.(0, sqrtdiagM), z))
	catch e
		-Inf
	end
	ℓ = energyjump + logpdf(ηpdist, θ) - logpdf(ηdist, θ)
	cost = 1 + cost1 + cost2
	grad_eval = grad_eval1 + grad_eval2 + grad_eval3
	if log(rand(rng)) ≤ ℓ
		return xp, min(0, ℓ), energyjump, cost, θ, grad_eval
	else
		return x, min(0, ℓ), 0.0, cost, θ, grad_eval
	end
end

function fix_step(x, f, θ0, target, sqrtdiagMhat, rng)
	z = rand(rng, auxtarget)
	xp, zp, grad_eval = f(x, z, θ0, target, sqrtdiagM)
	ℓ = LogDensityProblems.logdensity(target, xp) + sum(logpdf.(Normal.(0, sqrtdiagM), zp)) - LogDensityProblems.logdensity(target, x) - sum(logpdf.(Normal.(0, sqrtdiagM), z))
	cost = 1
	if log(rand(rng)) ≤ ℓ
		return xp, min(0, ℓ), ℓ, cost, θ0, grad_eval
	else
		return x, min(0, ℓ), 0.0, cost, θ0, grad_eval
	end
end

function run_sampler(x0, kernel, f, θ0, target, sqrtdiagMhat, niter, rng)
	x = copy(x0)
	xs = zeros(niter, length(x0))
	cs = zeros(niter)
	logas = zeros(niter)
	ejumps = zeros(niter)
	thetas = zeros(niter)
	grad_evals = zeros(niter)
	for i ∈ 1:niter
		x, logacc, ejump, cost, θ, grad_eval = kernel(x, f, θ0, target, sqrtdiagMhat, rng)
		xs[i, :] .= x
		cs[i] = (i == 1) ? cost : cs[i-1] + cost
		logas[i] = logacc
		ejumps[i] = ejump
		thetas[i] = θ
		grad_evals[i] = (i == 1) ? grad_eval : grad_evals[i-1] + grad_eval
	end
	return xs, cs, logas, ejumps, thetas, grad_evals
end


# run_sampler(zeros(100), auto_step, fMALA, 1.0, Funnel(99, 2.0), ones(100), 100)

# samples, stats_df = autostep2_sample_model("mixture", 1, "AutoStep MALA", 10, false)
# print(stats_df)
