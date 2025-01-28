using Pigeons, DataFrames, CSV, BridgeStan

pt = pigeons(
    target = Pigeons.stan_funnel(1, 0.1),
    variational = GaussianReference(first_tuning_round = 5),
    n_chains_variational = 10,
    record = [traces],
    n_rounds = 5)

samples = get_sample(pt)
samples = DataFrame(samples, :auto)
CSV.write("icml2025/samples/funnel2.csv", samples)



#plot(collect(df[1,:]), collect(df[2,:]))
df = CSV.read("icml2025/samples/funnel2.csv", DataFrame)