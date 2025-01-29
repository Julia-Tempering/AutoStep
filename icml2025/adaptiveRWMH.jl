using AdvancedMH, Distributions, MCMCChains, LogDensityProblems, LinearAlgebra
using CSV, DataFrames, DelimitedFiles, JSON, Turing, Random
include("utils.jl")

# using NUTS in Turing.jl
function adaptive_rwmh_sample_from_model(model, seed, n_rounds; max_samples = 2^25, kwargs...)
    # make model and data from the arguments
    my_data = stan_data(model)
    my_model = logdens_model(model, my_data)
    Random.seed!(seed)

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
    minKSess = min_KSess(samples, model)
    mean_1st_dim = mean(samples[1])
    var_1st_dim = var(samples[1])
    acceptance_prob = sum(1 for i in 2:n_samples if samples[i] != samples[i-1])/(n_samples - 1)
    energy_jump_dist = mean(abs.(diff(chain[:lp], dims=1)))
    stats_df = DataFrame(
        explorer = "adaptive RWMH", model = model, 
        mean_1st_dim = mean_1st_dim, var_1st_dim = var_1st_dim, time=my_time, jitter_std = 0.0, n_logprob = n_logprob, 
        n_steps=0, #zero gradient
        miness=miness, minKSess = minKSess, acceptance_prob=acceptance_prob, step_size=0.0, n_rounds = n_rounds, 
        energy_jump_dist = energy_jump_dist)
    return samples, stats_df
end