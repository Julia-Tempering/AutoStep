# include("../test/activate_test_env.jl")

include("adaptiveRWMH.jl")
include("adaptiveMALA.jl")
include("nuts.jl")
include("drhmc.jl")
include("autostep.jl")

models = ["funnel2"]
seeds = 1:30
n_rounds = 15
exp_results = DataFrame(
    explorer = String[],
    mean_1st_dim = Float64[],
    var_1st_dim = Float64[],
    time = Float64[],
    jitter_std = Float64[],
    n_logprob = Int[],
    n_steps = Int[],
    miness = Float64[],
    minKSess = Float64[],
    acceptance_prob = Float64[],
    step_size = Float64[],
    n_rounds = Int[],
    energy_jump_dist = Float64[]
)

for model in models
    for seed in seeds
        samples_rwmh, stats_rwmh = adaptive_rwmh_sample_from_model(model, seed, n_rounds)
        samples_mala, stats_mala = adaptive_mala_sample_from_model(model, seed, n_rounds)
        samples_nuts, stats_nuts = nuts_sample_from_model(model, seed, n_rounds)
        println("Current at seed $seed")
        samples_drhmc, stats_drhmc = drhmc_sample_from_model(model, seed, n_rounds)
        samples_autorwmh, stats_autorwmh = pt_sample_from_model(model, seed, "AutoStep RWMH", n_rounds)
        samples_automala, stats_automala = pt_sample_from_model(model, seed, "AutoStep MALA", n_rounds)
        samples_slicer, stats_slicer = pt_sample_from_model(model, seed, "HitAndRunSlicer", n_rounds)
        # save the samples from the first run
        if seed == 5
            CSV.write("icml2025/samples/$(model)_adaptive_rwmh.csv", DataFrame(samples_rwmh, :auto))
            CSV.write("icml2025/samples/$(model)_adaptive_mala.csv", DataFrame(samples_mala, :auto))
            CSV.write("icml2025/samples/$(model)_nuts.csv", DataFrame(samples_nuts, :auto))
            CSV.write("icml2025/samples/$(model)_drhmc.csv", DataFrame(samples_drhmc, :auto))
            CSV.write("icml2025/samples/$(model)_autorwmh.csv", DataFrame(samples_autorwmh, :auto))
            CSV.write("icml2025/samples/$(model)_automala.csv", DataFrame(samples_automala, :auto))
            CSV.write("icml2025/samples/$(model)_slicer.csv", DataFrame(samples_slicer, :auto))
        end
        # concatenate the dataframe of experiment results
        append!(exp_results, stats_rwmh)
        append!(exp_results, stats_mala)
        append!(exp_results, stats_nuts)
        append!(exp_results, stats_drhmc)
        append!(exp_results, stats_autorwmh)
        append!(exp_results, stats_automala)
        append!(exp_results, stats_slicer)
    end
end

CSV.write("icml2025/exp_results.csv", exp_results)
