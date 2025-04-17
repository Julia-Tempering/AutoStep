using Plots
using Distributions
using Statistics
using HypothesisTests
using Suppressor
import Base: rand
import Distributions: logpdf, gradlogpdf
include("autostep.jl")

struct FatStdNormal{T<:Real} <: ContinuousUnivariateDistribution
	width::T
end

function Base.rand(d::FatStdNormal)
	r = rand()
	if r ≤ 1/(1+d.width/sqrt(2*π))
		# sample from the gaussian part
		r = rand(Normal())
		if r ≤ 0
			return -d.width/2 + r
		else
			return d.width/2 + r
		end
	else
		#sample from the flat part
		return -d.width/2 + rand()*d.width
	end
end

function Distributions.logpdf(d::FatStdNormal, x::Real)
	if x < -d.width/2
		return logpdf(Normal(), x+d.width/2) - log(1+d.width/sqrt(2*π))
	elseif x > d.width/2
		return logpdf(Normal(), x-d.width/2) - log(1+d.width/sqrt(2*π))
	else
		return -log(sqrt(2*π)) - log(1+d.width/sqrt(2*π))
	end
end

function Distributions.gradlogpdf(d::FatStdNormal, x::Real)
	if x < -d.width/2
		return gradlogpdf(Normal(), x+d.width/2)
	elseif x ≤ d.width/2
		return 0
	else
		return gradlogpdf(Normal(), x-d.width/2)
	end
end

function Distributions.gradlogpdf(d::Laplace, x::Real)
	return -sign(x-d.μ)/d.θ
end

function Distributions.gradlogpdf(d::Cauchy{Float64}, x::Real)
	return -2*x/(1+x^2)
end


function run(f, target, θ0, log10norms, n_samp)
	auxtarget = Normal()
	initial = Normal()
    asyms = []
	syms = []
    for log10norm in log10norms
		nrm = (10.)^log10norm
		tsyms = []
		tasyms = []
		for i in 1:n_samp
			x = rand(target)
			x *= nrm/abs(x) 
        	xp, sym_loga, ejump, cost, θ, jittered = auto_step(x, f, θ0, target, auxtarget, true)
        	xp, asym_loga, ejump, cost, θ, jittered = auto_step(x, f, θ0, target, auxtarget, false)
			push!(tsyms, sym_loga)
			push!(tasyms, asym_loga)
		end
		push!(syms, mean(exp.(tsyms .- maximum(tsyms)))*exp(maximum(tsyms)))
		push!(asyms, mean(exp.(tasyms .- maximum(tasyms)))*exp(maximum(tasyms)))
	end
	return syms, asyms
end

function main()
	f, fn = fMALA, "MALA" 
	#f, fn = fRWMH, "RWMH"
	n_samp = 10000000
	θ0 = 1.0
	log10norms = -5:0.1:2
    targets = [(Normal(), "Normal", :solid), (Laplace(), "Laplace", :dash), (Cauchy(), "Cauchy", :dot)]
	p = plot()
	pal = palette(:tab10)
	donelegend = false
	fs = 12
	plotoptions = Dict(:guidefontsize => fs, :tickfontsize => fs, :legendfontsize => fs, :dpi => 600)
	for (target, tn, ls) in targets
		syms, asyms = run(f, target, θ0, log10norms, n_samp)

        if !donelegend
			asymlbl = "Asym. AutoStep $fn"
			symlbl = "Sym. AutoStep $fn"
			donelegend = true
		else
			asymlbl = ""
			symlbl = ""
		end

		plot!(p,(10.).^log10norms, syms, xscale=:log10, yscale=:log10, ylimits=(1e-6, 1.2), xticks=[(10.)^i for i in -5:2], linewidth=2, label=symlbl, color=pal[1], linestyle=ls; plotoptions...)
		plot!(p,(10.).^log10norms, asyms, xscale=:log10, yscale=:log10, ylimits=(1e-6, 1.2), linewidth=2, legend=:bottom, label=asymlbl, color=pal[2], linestyle=ls; plotoptions...)
		#hack to get line style legend
		plot!(p, [(10.).^log10norms[1]], [syms[1]], color=:black, linestyle=ls, label=tn, dpi=600)
	end
	xlabel!(p, "\$||x||\$")
	ylabel!(p, "Log Acceptance Probability")
	savefig(p, "sym_vs_asym_$fn.png")

end


main()
