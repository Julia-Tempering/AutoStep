using Pigeons, DataFrames, CSV, BridgeStan

#= pt = pigeons(
    target = Pigeons.stan_funnel(1, 0.1),
    variational = GaussianReference(first_tuning_round = 5),
    n_chains_variational = 10,
    record = [traces],
    n_rounds = 5)

samples = get_sample(pt)
samples = DataFrame(samples, :auto)
CSV.write("icml2025/samples/funnel2.csv", samples) =#

using Random, Distributions

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
CSV.write("icml2025/samples/funnel2.csv", samples)