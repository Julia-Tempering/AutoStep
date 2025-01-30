using Pigeons, DataFrames, CSV, BridgeStan
include(joinpath(dirname(dirname(pathof(Pigeons))), "test", "supporting", "postdb.jl"))
stan_example_path(name) = dirname(dirname(pathof(Pigeons))) * "/examples/$name"

pt = pigeons(
    # TODO: three remaining reference models: mRNA, ionosphere, orbital
    target = StanLogPotential(stan_example_path("stan/mRNA.stan"), "icml2025/data/mRNA.json"),
    #StanLogPotential("icml2025/data/horseshoe_logit.stan", "icml2025/data/ionosphere.json"),
    #orbital
    variational = GaussianReference(first_tuning_round = 5),
    n_chains_variational = 5,
    record = [traces],
    n_rounds = 18)

samples = get_sample(pt)
samples = DataFrame(samples, :auto)
CSV.write("icml2025/samples/mRNA.csv", samples)

#= using Random, Distributions

function sample_funnel(N, scale)
    samples = Matrix{Float64}(undef, 2, N)  # Store samples as a 2×N matrix

    for i in 1:N
        v = rand(Normal(0, 3))
        x = rand(Normal(0, exp(v / scale)))  # Sample x from N(0, exp(scale * v))
        samples[:, i] = [v, x]  # Store in matrix
    end

    return samples
end

N = 10000000
scale = 0.6
samples = sample_funnel(N, scale)
samples = DataFrame(samples, :auto)
CSV.write("icml2025/samples/funnel2.csv", samples) =#