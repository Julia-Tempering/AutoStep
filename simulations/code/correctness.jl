using Plots
using Distributions
using Statistics
using HypothesisTests
using Suppressor
import Base: rand
import Distributions: logpdf, gradlogpdf
include("autostep.jl")

function Distributions.gradlogpdf(d::Laplace, x::Real)
	return -sign(x-d.μ)/d.θ
end

function Distributions.gradlogpdf(d::Cauchy{Float64}, x::Real)
	return -2*x/(1+x^2)
end

function run(f, target, θ0, n_samp)
	auxtarget = Normal()
	exact_xs = [rand(target) for i in 1:n_samp]
	mcmc_xs = []
	for n in 1:n_samp
		x = copy(exact_xs[n])
		for k in 1:100
			x, sym_loga, ejump, cost, θ, jittered = auto_step(x, f, θ0, target, auxtarget)
		end
		push!(mcmc_xs, x)
	end
	return ApproximateTwoSampleKSTest(Vector{Float64}(mcmc_xs), exact_xs)
end

function main()
	n_samp = 1000000
	θ0 = 1.0
    targets = [(Normal(), "Normal", :solid), (Laplace(), "Laplace", :dash), (Cauchy(), "Cauchy", :dot)]
	for (target, tn, ls) in targets
		ksmala = run(fMALA, target, θ0, n_samp)
		ksrwmh = run(fRWMH, target, θ0, n_samp)
		println("MALA Target $target p-value $(pvalue(ksmala))")
		println("RWMH Target $target p-value $(pvalue(ksrwmh))")
	end
end


main()
