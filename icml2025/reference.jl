using Pigeons, DataFrames, CSV, BridgeStan
include(joinpath(dirname(dirname(pathof(Pigeons))), "test", "supporting", "postdb.jl"))

pt = pigeons(
    target = StanLogPotential("icml2025/data/horseshoe_logit.stan", "icml2025/data/ionosphere.json"),
    variational = GaussianReference(first_tuning_round = 5),
    n_chains_variational = 5,
    record = [traces],
    n_rounds = 16)

samples = get_sample(pt)
samples = DataFrame(samples, :auto)
CSV.write("icml2025/samples/prostate.csv", samples)

#= using Random, Distributions

function sample_funnel(N, scale)
    samples = Matrix{Float64}(undef, 2, N)  # Store samples as a 2×N matrix

    for i in 1:N
        v = rand(Normal(0, 3))  # Sample v from N(0,1)
        x = rand(Normal(0, exp(0.5 * v / scale)))  # Sample x from N(0, exp(scale * v))
        samples[:, i] = [v, x]  # Store in matrix
    end

    return samples
end

N = 10000000
scale = 0.3
samples = sample_funnel(N, scale)
samples = DataFrame(samples, :auto)
CSV.write("icml2025/samples/funnel2.csv", samples) =#