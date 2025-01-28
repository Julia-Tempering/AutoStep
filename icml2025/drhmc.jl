include("utils.jl")
using Distributions, LinearAlgebra, ForwardDiff, MCMCChains
using Plots, Random


# Combined Delayed Rejection HMC sampling function
function dr_hmc(n_samples, epsilon, n_leapfrogs, max_proposals, reduction_factor, M, log_density_q, dim; initial_params = nothing)
	samples = Matrix{Float64}(undef, n_samples, dim)
	q = initial_params == nothing ? zeros(dim) : initial_params
	p = rand(MvNormal(zeros(dim), M)) # initial momentum
	n_halvings = zeros(n_samples)
	is_accept = zeros(n_samples)
	log_densities = zeros(n_samples)
	grad_eval = 0
	logprob_eval = 0

	for i in 1:n_samples
		# refresh momentum
		p = rand(MvNormal(zeros(dim), M))
		accepted = false
		n_halving = -1

		for k in 1:max_proposals
			# Adjust step size for each retry
			epsilon_k = epsilon / reduction_factor^(k - 1)
			n_halving += 1

			# perform multiple leapfrog steps
			q_proposed, p_proposed = leapfrog(q, p, epsilon_k, n_leapfrogs, log_density_q)
            if !all(isfinite, p_proposed)
                break
            end
			alpha_k = acceptance_prob(q, p, q_proposed, p_proposed, log_density_q, M, k, epsilon, reduction_factor, n_leapfrogs)
			logprob_eval += 4 * (k - 1)
			grad_eval += n_leapfrogs * (1 + 2 * (k - 1))

			if rand() < alpha_k
				q = q_proposed
				p = -p_proposed
				accepted = true
				break
			end
		end

		samples[i, :] = q
		n_halvings[i] = n_halving
		is_accept[i] = accepted
		log_densities[i] = log_density_q(q)
	end
    println("Done!")
	return samples, log_densities, mean(n_halvings), mean(is_accept), grad_eval, logprob_eval
end

# Leapfrog integrator for Hamiltonian dynamics with error handling
function leapfrog(q, p, epsilon, n_leapfrogs, log_density_q)
    q_new, p_new = q, p
    try
        for i in 1:n_leapfrogs
            p_new = p_new + 0.5 * epsilon .* ForwardDiff.gradient(log_density_q, q_new)
            q_new = q_new + epsilon .* p_new
            p_new = p_new + 0.5 * epsilon .* ForwardDiff.gradient(log_density_q, q_new)
        end
        return q_new, -p_new
    catch e
        # If an error occurs, return the original state
        return q, p
    end
end

# Delayed rejection acceptance probability
function acceptance_prob(q, p, q_pr, p_pr, log_density_q, M, k, epsilon, reduction_factor, n_leapfrogs)
	# Initial acceptance ratio without the correction factor
	π_q = log_density_q(q)
	π_q_pr = log_density_q(q_pr)
	norm_p = logpdf(MvNormal(zeros(length(p)), M), p)
	norm_p_pr = logpdf(MvNormal(zeros(length(p_pr)), M), p_pr)

	alpha = exp(π_q_pr + norm_p_pr - π_q - norm_p)

	# Correction factor from previous rejections up to k-1
	for i in 1:(k-1)
		# Generate q_gh and p_gh for the current proposal
		q_gh, p_gh = leapfrog(q_pr, p_pr, epsilon / reduction_factor^(i - 1), n_leapfrogs, log_density_q(q))

		# Generate q_rej and p_rej for the rejection step
		q_rej, p_rej = leapfrog(q, p, epsilon / reduction_factor^(i - 1), n_leapfrogs, log_density_q(q))

		# Compute acceptance probability for (q_gh, p_gh)
		alpha_gh_num = log_density_q(q_gh) + logpdf(MvNormal(zeros(length(p)), M), p_gh)
		alpha_gh_den = log_density_q(q_pr) + logpdf(MvNormal(zeros(length(p_pr)), M), p_pr)
		alpha_gh = min(1.0, exp(alpha_gh_num - alpha_gh_den))

		# Compute acceptance probability for (q_rej, p_rej)
		alpha_rej_num = log_density_q(q_rej) + logpdf(MvNormal(zeros(length(p)), M), p_rej)
		alpha_rej_den = π_q + norm_p
		alpha_rej = min(1.0, exp(alpha_rej_num - alpha_rej_den))

		# Update α with the correction factor
		alpha *= if alpha_rej == 1
			1
		else
			(1 - alpha_gh) / (1 - alpha_rej)
		end
	end

	return min(1.0, alpha)
end


function drhmc_sample_from_model(model, seed, n_rounds; max_samples = 2^25, kwargs...)
	# make model and data from the arguments
	my_data = stan_data(model)
	my_model = logdens_model(model, my_data)
    log_density_q(x) = LogDensityProblems.logdensity(my_model, x)
    dim = Int(my_data["dim"])
    # initialize DRHMC
    step_size = 0.1      # initial step size
    n_leapfrogs = 10     # Number of leapfrog steps per proposal(only in DRHMC)
    k_retries = 10       # Maximum number of retries (delayed rejection)
    a_factor = 2.0       # Step size reduction factor for retries
    M = Diagonal(ones(dim))    # Mass matrix

	Random.seed!(seed)

	n_samples = 2
	n_logprob = n_steps = 0
	miness = my_time = 0.0
    initial_params = nothing

	local samples
	local log_densities
    local acceptance_prob
	for i in 1:n_rounds
        n_samples = 2^i
		my_time += @elapsed samples, log_densities, halvings, acceptance_prob, grad_eval, logprob_eval = 
            dr_hmc(n_samples, step_size, n_leapfrogs, k_retries, a_factor, M, log_density_q, dim)
        println(my_time)
		n_steps += grad_eval # one leapfrog per iteration, one grad evaluation per leapfrog
		n_logprob += logprob_eval # one for proposal, one for current state
        samples = [samples[i, :] for i in 1:size(samples, 1)]
		miness = min_ess_all_methods(samples, model)
        initial_params = samples[end]
	end
	mean_1st_dim = mean(samples[1])
	var_1st_dim = var(samples[1])
    energy_jump_dist = mean(abs.(diff(log_densities)))
	stats_df = DataFrame(
		mean_1st_dim = mean_1st_dim, var_1st_dim = var_1st_dim, time = my_time, jitter_std = 0, n_logprob = n_logprob, n_steps = n_steps,
		miness = miness, acceptance_prob = acceptance_prob, step_size = step_size, n_rounds = n_rounds, energy_jump_dist = energy_jump_dist)
	return samples, stats_df
end

drhmc_sample_from_model("funnel2", 1, 10)



#= # Run DR-HMC sampler with Neal's Funnel
model = "funnel2"
my_data = stan_data(model)
my_model = logdens_model(model, my_data)
log_density_q(x) = LogDensityProblems.logdensity(my_model, x)
n_samples = 5000
epsilon = 0.05             # Initial step size
dim_x = my_data["dim"]
n_steps = 10         # Number of leapfrog steps per proposal(only in DRHMC)
k_retries = 10       # Maximum number of retries (delayed rejection)
a_factor = 2.0       # Step size reduction factor for retries
M = Diagonal(ones(Int(dim_x + 1)))      # Mass matrix

Random.seed!(1)
# sample from DR HMC
samples, log_densities, halvings, acceptance_rate, grad_eval, logprob_eval = dr_hmc(n_samples, epsilon, n_steps, k_retries, a_factor, M, log_density_q, 2) =#
