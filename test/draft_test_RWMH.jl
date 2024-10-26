using Pigeons
using BridgeStan
using AutoStep
using MCMCChains
using Distributions
using Statistics
using DataFrames
using CSV
using StatsPlots

function string_to_model(model) 
    if model == "logearn_height" # dim = 3
        ms = Pigeons.StanLogPotential("data/logearn_height_model.stan", "data/heights.json")
    elseif model == "eight_schools_centered" # dim = 10
        ms = Pigeons.stan_eight_schools()
    elseif model == "earn_height" # dim = 3
        ms = Pigeons.StanLogPotential("data/earn_height_model.stan", "data/heights.json")
#    elseif model == "lotka_volterra"
#        ms = lotka_volterra_model()
    elseif model == "hmm" # dim = 104
        ms = Pigeons.StanLogPotential("data/hmm_model.stan", "data/hmm.json")
    elseif model == "normal"
        ms = toy_mvn_target(50)
    elseif model == "funnel"
        ms = Pigeons.stan_funnel(2)
    elseif model == "banana"
        ms = Pigeons.stan_banana(2)
    end 
    return ms
end

function string_to_explorer(explorer)
    if explorer == "autoRWMH" 
        ex = SimpleRWMH(step_size_selector = autoRWMH.MHSelector(), step_jitter_dist = Dirac(0.0))
    elseif explorer == "soft_autoRWMH" 
        ex = SimpleRWMH(step_size_selector = autoRWMH.MHSelector())
    elseif explorer == "autoRWMH_inverted" 
        ex = SimpleRWMH(step_jitter_dist = Dirac(0.0))
    elseif explorer == "soft_autoRWMH_inverted" 
        ex = SimpleRWMH()
    elseif explorer == "slice" 
        ex = SliceSampler()
    elseif explorer == "autoMALA" 
        ex = AutoMALA()
    end
end


function main()
    # simulation settings
    seeds = [1]#,2,3,4,5,6,7,8,9,10]
    models = ["eight_schools_centered"]
    explorers = ["slice"]#["autoRWMH", "soft_autoRWMH", "autoRWMH_inverted", "soft_autoRWMH_inverted", "slice", "autoMALA"]
    n_rounds = 15
    n_chains = 10 #?????
    result = DataFrame(explorer = String[], model = String[], minESS = Float64[],
        minESSperSec = Float64[])

    # simulation
    for model in models
        for explorer in explorers 
            for seed in seeds
                pt = pigeons(
                    target = string_to_model(model),
                    # reference = model,
                    seed = seed, 
                    explorer = string_to_explorer(explorer),
                    record = [traces; record_default(); round_trip],
                    n_rounds = n_rounds,
                    n_chains = n_chains)
                print(pt.shared.reports.summary)
                display(StatsPlots.plot(Chains(pt)))
                ess_df = MCMCChains.ess(Chains(pt))
                time = sum(pt.shared.reports.summary.last_round_max_time)
                push!(result, (explorer, model, minimum(ess_df.nt.ess),
                minimum(ess_df.nt.ess ./ time)))
            end
        end 
    end
    # Remove rows with any NaNs
    # clean_results_ESS = filter(row->!(isnan(row.minESS)), result)
    CSV.write("results/InferHub/eight_schools_ess.csv", result)
end 

main()
