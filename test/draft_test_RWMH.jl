using Pigeons
using BridgeStan
using autoRWMH
using MCMCChains
using Distributions
using Statistics
using DataFrames
using CSV

struct MyLogPotential
    horizontal_spread::Float64
    vertical_spread::Float64
end

function (log_potential::MyLogPotential)(x)
    p1, p2 = x
    return logpdf(Normal(0.0, log_potential.horizontal_spread), p1) + 
    logpdf(Normal(0.0, exp(p1 / log_potential.vertical_spread)), p2)
end


my_log_potential = MyLogPotential(3.0, 2.0)
my_log_potential([0.5, 0.5])
Pigeons.initialization(::MyLogPotential, ::AbstractRNG, ::Int) = [1.0, 1.0]

function main()
    # simulation settings
    seeds = [1,2,3,4,5,6,7,8,9,10]
    models = [MyLogPotential(3,2)] #[toy_mvn_target(5)]
    explorer = [SimpleRWMH()]
    n_rounds = 10
    result = DataFrame(explorer = String[], model = String[], minESS = Float64[],
        minESSperSec = Float64[])

    # simulation
    for model in models
        for seed in seeds 
            pt = pigeons(
                target = model,
                reference = model,
                seed = seed, 
                explorer = SimpleRWMH(),
                record = [traces; record_default()],
                n_rounds = n_rounds)
            ess_df = MCMCChains.ess(Chains(pt))
            time = sum(pt.shared.reports.summary.last_round_max_time)
            push!(result, ("softRWMH_inverted", "normal5", minimum(ess_df.nt.ess),
            minimum(ess_df.nt.ess ./ time)))
        end 
    end
    # Remove rows with any NaNs
    # clean_results_ESS = filter(row->!(isnan(row.minESS)), result)
    # Plotting boxplots
    CSV.write("results/softrwmh_inv_normal_ess.csv", result)
end 

main()
