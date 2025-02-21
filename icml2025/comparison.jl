include("adaptiveRWMH.jl")
include("adaptiveMALA.jl")
include("nuts.jl")
include("drhmc.jl")
include("autostep.jl")
include("autostep2.jl")

models = ["funnel2"]
seeds = [1]
n_rounds = 15
exp_results = DataFrame(
    explorer = String[],
    model = String[],
    seed = Int[], 
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
        # samples_autorwmh_precond, stats_autorwmh_precond = autostep2_sample_model(model, seed, "AutoStep RWMH (precond)", n_rounds, true)
        # CSV.write("icml2025/temp/$(seed)_$(model)_autorwmh_precond.csv", DataFrame(samples_autorwmh_precond, :auto))
        # append!(exp_results, stats_autorwmh_precond)
        # samples_autorwmh_precond = nothing

        # samples_automala_precond, stats_automala_precond = autostep2_sample_model(model, seed, "AutoStep MALA (precond)", n_rounds, true)
        # CSV.write("icml2025/temp/$(seed)_$(model)_automala_precond.csv", DataFrame(samples_automala_precond, :auto))
        # append!(exp_results, stats_automala_precond)
        # samples_automala_precond = nothing

        samples_slicer, stats_slicer = pt_sample_from_model(model, seed, "HitAndRunSlicer", n_rounds)
        CSV.write("icml2025/temp/$(seed)_$(model)_slicer.csv", DataFrame(samples_slicer, :auto))
        append!(exp_results, stats_slicer)
        samples_slicer = nothing

        samples_autorwmh, stats_autorwmh = autostep2_sample_model(model, seed, "AutoStep RWMH", n_rounds, false)
        CSV.write("icml2025/temp/$(seed)_$(model)_autorwmh.csv", DataFrame(samples_autorwmh, :auto))
        append!(exp_results, stats_autorwmh)
        samples_autorwmh = nothing

        samples_rwmh, stats_rwmh = adaptive_rwmh_sample_from_model(model, seed, n_rounds)
        CSV.write("icml2025/temp/$(seed)_$(model)_adaptive_rwmh.csv", DataFrame(samples_rwmh, :auto))
        append!(exp_results, stats_rwmh)
        samples_rwmh = nothing

        samples_automala, stats_automala = autostep2_sample_model(model, seed, "AutoStep MALA", n_rounds, false)
        CSV.write("icml2025/temp/$(seed)_$(model)_automala.csv", DataFrame(samples_automala, :auto))
        append!(exp_results, stats_automala)
        samples_automala = nothing

        samples_mala, stats_mala = adaptive_mala_sample_from_model(model, seed, n_rounds)
        CSV.write("icml2025/temp/$(seed)_$(model)_adaptive_mala.csv", DataFrame(samples_mala, :auto))
        append!(exp_results, stats_mala)
        samples_mala = nothing

        samples_nuts, stats_nuts = nuts_sample_from_model(model, seed, n_rounds)
        CSV.write("icml2025/temp/$(seed)_$(model)_nuts.csv", DataFrame(samples_nuts, :auto))
        append!(exp_results, stats_nuts)
        samples_nuts = nothing

        samples_drhmc, stats_drhmc = drhmc_sample_from_model(model, seed, n_rounds)
        CSV.write("icml2025/temp/$(seed)_$(model)_drhmc.csv", DataFrame(samples_drhmc, :auto))
        append!(exp_results, stats_drhmc)
        samples_drhmc = nothing
        
        println("Finish seed $seed")
        CSV.write("icml2025/exp_results_$(model)_temp.csv", exp_results)
    end
end


