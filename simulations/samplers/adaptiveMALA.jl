using AdvancedHMC, Distributions, LogDensityProblems, LinearAlgebra
using CSV, DataFrames, DelimitedFiles, JSON, Turing, ReverseDiff
include("utils.jl")

function adaptive_mala_sample_from_model(model, seed, n_rounds; max_samples = 2^25, kwargs...)
	# make model and data from the arguments
	my_data = stan_data(model)
	my_model = logdens_model(model, my_data)
	Random.seed!(seed)

	n_samples = 2
	n_logprob = n_steps = 0
	miness = my_time = 0.0
	# initialize adaptive MALA kernel
	n_adapts = 0
    step_size = 0.1
    dim = LogDensityProblems.dimension(my_model)
	metric = DenseEuclideanMetric(dim)
	hamiltonian = Hamiltonian(metric, my_model, ReverseDiff)
	integrator = Leapfrog(step_size)
	kernel = HMCKernel(Trajectory{EndPointTS}(integrator, FixedNSteps(1))) # n_leapfrog = 1 to recover MALA
	adaptor = MassMatrixAdaptor(metric)
	hmc = HMCSampler(kernel, metric, adaptor)
    initial_params = nothing

	local samples
	local stats
	for i in 1:n_rounds
        n_samples = 2^i
		my_time += @elapsed samples, stats = sample(hamiltonian, kernel, zeros(dim), n_samples, adaptor, n_adapts; 
            progress = true) #, initial_params = initial_params)
		n_steps += n_samples # one leapfrog per iteration, one grad evaluation per leapfrog
		n_logprob += 2 * n_samples # one for proposal, one for current state
        # initial_params = samples[end]
	end
    miness = min_ess_all_methods(samples, model)
    # minKSess = min_KSess(samples, model)
	mean_1st_dim = mean(samples[1])
	var_1st_dim = var(samples[1])
	acceptance_prob = sum(stat.is_accept for stat in stats) / n_samples
    energy_jump_dist = mean(abs.(diff([stat.log_density for stat in stats])))
	stats_df = DataFrame(
        explorer = "adaptive MALA", model = model, seed = seed, 
		mean_1st_dim = mean_1st_dim, var_1st_dim = var_1st_dim, time = my_time, jitter_std = 0.0, 
        n_logprob = n_logprob, n_steps = n_steps,
		miness = miness, minKSess = 0.0, acceptance_prob = acceptance_prob, step_size = step_size, 
        n_rounds = n_rounds, energy_jump_dist = energy_jump_dist)
	return samples, stats_df
end

# adaptive_mala_sample_from_model("funnel2", 1, 15)
