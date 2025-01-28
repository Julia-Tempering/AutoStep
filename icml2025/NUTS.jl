using AdvancedHMC, Distributions, MCMCChains, LogDensityProblems, LinearAlgebra
using CSV, DataFrames, DelimitedFiles, JSON, Random, ForwardDiff
include("utils.jl")

# using NUTS in Turing.jl
function nuts_sample_from_model(model, seed, n_rounds; max_samples = 2^25, kwargs...)
    # make model and data from the arguments
    my_data = stan_data(model; kwargs...)
    my_model = turing_nuts_model(model, my_data)
    Random.seed!(seed)

    # run until minESS threshold is reached
    n_samples = 2
    n_logprob = n_steps = 0
    miness = my_time = 0.0
    initial_params = nothing
    local samples
    local chain
    for i in 1:n_rounds
        n_samples = 2^i
        my_time += @elapsed chain = sample(my_model, AdvancedHMC.NUTS(0.65), n_samples; # target acceptance = 0.65
            chain_type = Chains, initial_params = initial_params)
        n_steps += sum(chain[:n_steps]) # count leapfrogs not including warmup
        n_logprob += 2*n_samples # one for the proposal, one for the current state
        samples = [chain[param] for param in names(chain)[1:end-13]] # discard 12 aux vars
        samples = [vec(sample) for sample in samples] # convert to vectors
        samples = [collect(row) for row in eachrow(hcat(samples...))] # convert to format compatible with min_ess_all_methods
        miness = min_ess_all_methods(samples, model)
        initial_params = samples[end]
    end
    mean_1st_dim = mean(samples[1])
    var_1st_dim = var(samples[1])
    acceptance_prob = mean(chain[:acceptance_rate])
    step_size = mean(chain[:step_size])
    energy_jump_dist = mean(abs.(diff(chain[:log_density], dims=1)))
    stats_df = DataFrame(
        mean_1st_dim = mean_1st_dim, var_1st_dim = var_1st_dim, time=my_time, jitter_std = 0, n_logprob = n_logprob, n_steps=n_steps, 
        miness=miness, acceptance_prob=acceptance_prob, step_size=step_size, n_rounds = n_rounds, energy_jump_dist = energy_jump_dist)
    return samples, stats_df
end

nuts_sample_from_model("funnel2", 1, 15)