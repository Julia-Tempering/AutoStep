using Pigeons, DataFrames, CSV, BridgeStan, LogDensityProblems, Random, LinearAlgebra, Statistics
include("orbital_model_definition.jl")
include(joinpath(dirname(dirname(pathof(Pigeons))), "test", "supporting", "postdb.jl"))

function rwmh(x, ℓ, step, logπ, sqrtΣ)
	xp = x + step*(sqrtΣ*randn(length(x)))
	ℓp = LogDensityProblems.logdensity(logπ, xp)
	if log(rand()) <= ℓp - ℓ
		return xp, ℓp, true
	else
		return x, ℓ, false
	end
end

function mcmc(x0, logπ, niter, steps, skip, name)
	xs = zeros(niter, length(x0))
	ℓs = zeros(niter)
	accs = zeros(length(steps))
	naccs = 0
	sqrtΣ = I
	x = copy(x0)
	ℓ = LogDensityProblems.logdensity(logπ, x)
	t0 = time()
	for i=1:niter
		for k = 1:skip
			for j = 1:length(steps)
				x, ℓ, acc = rwmh(x, ℓ, steps[j], logπ, sqrtΣ)
				accs[j] += Int(acc)
			end
			naccs += 1
		end
		xs[i, :] .= x
		ℓs[i] = ℓ
		if ispow2(i)
			tcur = time()
			iters_per_sec = i/(tcur-t0)
			mins_remaining = (niter - i)/iters_per_sec/60.0
			println("Iteration $i / $niter ItersPerSec $(round(iters_per_sec, digits=2)) MinsRemain $(round(mins_remaining, digits=2)) Log10Steps $(round.(log10.(steps), sigdigits=2)) Avg Accs $(round.(accs/naccs, sigdigits=2))")
    		df = DataFrame([xs[j, :] for j = 1:i], :auto)
			CSV.write("icml2025/samples_simple/$name.csv", df)
			if i > 2*length(x)
				sqrtΣ = cholesky(cov(xs[Int(i/2):i, :]) + 1e-6*I).L
			end
		end
	end
    df = DataFrame([xs[i, :] for i = 1:size(xs, 1)], :auto)
	CSV.write("icml2025/samples_simple/$name.csv", df)
	return xs, ℓs, accs
end

function main()
	include(joinpath(dirname(dirname(pathof(Pigeons))), "test", "supporting", "postdb.jl"))
	stan_example_path(name) = dirname(dirname(pathof(Pigeons))) * "/examples/$name"
	orbital_target = Octofitter.LogDensityModel(GL229A; autodiff=:ForwardDiff, verbosity=4)
	mrna_target = StanLogPotential(stan_example_path("stan/mRNA.stan"), "icml2025/data/mRNA.json")
	ionosphere_target = StanLogPotential("icml2025/data/horseshoe_logit.stan", "icml2025/data/ionosphere.json")
	sonar_target = StanLogPotential("icml2025/data/horseshoe_logit.stan", "icml2025/data/sonar.json")
	prostate_target = StanLogPotential("icml2025/data/horseshoe_logit.stan", "icml2025/data/prostate.json")
	kilp_target = log_potential_from_posterior_db("kilpisjarvi_mod-kilpisjarvi.json")

	##MRNA
	#x0 = zeros(LogDensityProblems.dimension(mrna_target))
	#mcmc(x0, mrna_target, 10000000, (10.).^(-4:0.5:1), 10, "mRNA")

    ##Orbital
	#x0 = zeros(LogDensityProblems.dimension(orbital_target))
	#mcmc(x0, orbital_target, 10000000, (10.).^(-6:0.5:1), 10, "orbital")

	##Ionosphere
	#x0 = zeros(LogDensityProblems.dimension(ionosphere_target))
	#mcmc(x0, ionosphere_target, 10000000, (10.).^(-3:0.5:0), 10, "ionosphere")

    #Sonar
	x0 = zeros(LogDensityProblems.dimension(sonar_target))
	mcmc(x0, sonar_target, 10000000, (10.).^(-3:0.5:0), 10, "sonar")

    ##Prostate
	#x0 = zeros(LogDensityProblems.dimension(prostate_target))
	#mcmc(x0, prostate_target, 10000000, (10.).^(-3:0.5:0), 10, "prostate")

    ##Kilpisjarvi
	#x0 = zeros(LogDensityProblems.dimension(kilp_target))
	#mcmc(x0, kilp_target, 10000000, (10.).^(-2:0.5:2), 10, "kilpisjarvi")

end

main()




