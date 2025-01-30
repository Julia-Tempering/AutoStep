include("adaptiveRWMH.jl")
include("adaptiveMALA.jl")
include("nuts.jl")
include("drhmc.jl")
include("autostep.jl")
include("autostep2.jl")

models = ["funnel100"]
seeds = [1]
n_rounds = 14
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
        #= samples_autorwmh_precond, stats_autorwmh_precond = autostep2_sample_model(model, seed, "AutoStep RWMH (precond)", n_rounds, true)
        CSV.write("icml2025/temp/$(seed)_$(model)_adaptive_rwmh_precond.csv", DataFrame(samples_autorwmh_precond, :auto))
        append!(exp_results, stats_autorwmh_precond)
        samples_autorwmh_precond = nothing
        stats_autorwmh_precond = nothing
        GC.gc()

        samples_automala_precond, stats_automala_precond = autostep2_sample_model(model, seed, "AutoStep MALA (precond)", n_rounds, true)
        CSV.write("icml2025/temp/$(seed)_$(model)_adaptive_mala_precond.csv", DataFrame(samples_automala_precond, :auto))
        append!(exp_results, stats_automala_precond)
        samples_automala_precond = nothing
        stats_automala_precond = nothing
        GC.gc() =#

        samples_autorwmh, stats_autorwmh = autostep2_sample_model(model, seed, "AutoStep RWMH", n_rounds, false)
        CSV.write("icml2025/temp/$(seed)_$(model)_autorwmh.csv", DataFrame(samples_autorwmh, :auto))
        append!(exp_results, stats_autorwmh)
        samples_autorwmh = nothing
        print(stats_autorwmh)

        samples_automala, stats_automala = autostep2_sample_model(model, seed, "AutoStep MALA", n_rounds, false)
        CSV.write("icml2025/temp/$(seed)_$(model)_automala.csv", DataFrame(samples_automala, :auto))
        append!(exp_results, stats_automala)
        samples_automala = nothing
        print(stats_automala)

        samples_rwmh, stats_rwmh = adaptive_rwmh_sample_from_model(model, seed, n_rounds)
        CSV.write("icml2025/temp/$(seed)_$(model)_adaptive_rwmh.csv", DataFrame(samples_rwmh, :auto))
        append!(exp_results, stats_rwmh)
        samples_rwmh = nothing
        print(stats_rwmh)

        samples_mala, stats_mala = adaptive_mala_sample_from_model(model, seed, n_rounds)
        CSV.write("icml2025/temp/$(seed)_$(model)_adaptive_mala.csv", DataFrame(samples_mala, :auto))
        append!(exp_results, stats_mala)
        samples_mala = nothing
        print(stats_mala)

        samples_nuts, stats_nuts = nuts_sample_from_model(model, seed, n_rounds)
        CSV.write("icml2025/temp/$(seed)_$(model)_nuts.csv", DataFrame(samples_nuts, :auto))
        append!(exp_results, stats_nuts)
        samples_nuts = nothing
        print(stats_nuts)

        println("Current at seed $seed")
        # samples_drhmc, stats_drhmc = drhmc_sample_from_model(model, seed, n_rounds)
        # CSV.write("icml2025/temp/$(seed)_$(model)_drhmc.csv", DataFrame(samples_drhmc, :auto))
        # append!(exp_results, stats_drhmc)
        # samples_drhmc = nothing
        # print(stats_drhmc)

        # samples_slicer, stats_slicer = pt_sample_from_model(model, seed, "HitAndRunSlicer", n_rounds)
        # CSV.write("icml2025/temp/$(seed)_$(model)_slicer.csv", DataFrame(samples_slicer, :auto))
        # append!(exp_results, stats_slicer)
        # samples_slicer = nothing
        # print(stats_slicer)
    end
    CSV.write("icml2025/exp_results.csv", exp_results)
end


