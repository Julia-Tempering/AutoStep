using Plots
using Distributions
using Statistics
using HypothesisTests
using Suppressor
include("autostep.jl")

function Distributions.gradlogpdf(d::Laplace, x::Real)
	return -sign(x-d.μ)/d.θ
end

function Distributions.gradlogpdf(d::Cauchy{Float64}, x::Real)
	return -2*x/(1+x^2)
end

# if x is truly iid, and F continuous, one expects the KS statistic sqrt(n)D_n to converge to the kolmogorov distribution.
# Most of its mass is concentrated around 1.
function KSess(xs, target)
	N = length(xs)
	B = Int(ceil(sqrt(N)))
	j = 0
	i = 1
	s = 0
	while i < N
		res = @suppress ExactOneSampleKSTest(xs[i:min(N, i+B)], target)
		s += res.δ
		j += 1
		i = min(N, i+B) 
	end
	s /= j
	return s^(-4)
end
#	return (res.δ)^(-2)

# taken from the autostep experiments repo from batchmeans
function ess(samples, target)
	posterior_mean = mean(target)
	posterior_sd = std(target)
    n_samples = length(samples)
    n_blocks  = 1 + isqrt(n_samples)
    blk_size  = n_samples ÷ n_blocks # takes floor of division
    centered_batch_means = map(1:n_blocks) do b
        i_start = blk_size*(b-1) + 1
        i_end   = blk_size*b
        mean(x -> (x - posterior_mean)/posterior_sd, @view samples[i_start:i_end])
    end
    n_blocks / mean(abs2, centered_batch_means)
end

#function run(f, target, log10θ0s, n_samp)
#	auxtarget = Normal()
#    asyms = []
#	syms = []
#    for log10θ0 in log10θ0s
#		θ0 = (10.)^log10θ0
#		for i in 1:n_samp
#			x = rand(target)
#			rngstate = copy(Random.default_rng())
#        	xp, loga, ejump, cost, jittered = auto_step(x, f, θ0, target, auxtarget)
#			copy!(Random.default_rng(), rngstate)
#        	xp, jitloga, jitejump, jitcost, jittered = auto_step_jitter(x, f, θ0, target, auxtarget)
#		end
#	end
#	return 0
#end

function run(f, target, logθ0s, niter)
	auxtarget = Normal(0., 1.)
	initial = target	
	auto_log_acceptance_ratio = []
	auto_jump = []
	auto_ejump = []
	auto_ess = []
	auto_KSess = []
	auto_cost = []
	auto_jit = []
    jitauto_log_acceptance_ratio = []
    jitauto_jump = []
    jitauto_ejump = []
	jitauto_ess = []
	jitauto_KSess = []
	jitauto_cost = []
	jitauto_jit = []
	for logθ0 in logθ0s
		θ0 = (10.)^logθ0
		# autostep
		xs, cs, logas, ejumps, jitters = mcmc(auto_step, f, θ0, target, auxtarget, initial, niter)
		maxlogas = maximum(logas)
		logas .-= maxlogas
		push!(auto_log_acceptance_ratio, maxlogas + log(mean(exp.(logas))))
		push!(auto_jump, mean(abs.(xs[2:end] .- xs[1:end-1])))
		push!(auto_ess, ess(xs, target))
		push!(auto_KSess, KSess(xs, target))
		push!(auto_cost, cs[end])
		push!(auto_ejump, mean(abs.(ejumps)))
		push!(auto_jit, mean(jitters))
			
		# jitautostep
        xs, cs, logas, ejumps, jitters = mcmc(auto_step_jitter, f, θ0, target, auxtarget, initial, niter)
        maxlogas = maximum(logas)
		logas .-= maxlogas
		push!(jitauto_log_acceptance_ratio, maxlogas + log(mean(exp.(logas))))
		push!(jitauto_jump, mean(abs.(xs[2:end] .- xs[1:end-1])))
		push!(jitauto_ess, ess(xs, target))
		push!(jitauto_KSess, KSess(xs, target))
		push!(jitauto_cost, cs[end])
		push!(jitauto_ejump, mean(abs.(ejumps)))
		push!(jitauto_jit, mean(jitters))

	end	
	return auto_log_acceptance_ratio, auto_jump, auto_ess, auto_KSess, auto_cost, auto_ejump, auto_jit,
			jitauto_log_acceptance_ratio, jitauto_jump, jitauto_ess, jitauto_KSess, jitauto_cost, jitauto_ejump, jitauto_jit
end


function main()
	minlogθ0 = -1
	maxlogθ0 = 1
	Nθs = 100
	logθ0s = (Nθs .- Array(0:Nθs))/Nθs*minlogθ0 + Array(0:Nθs)/Nθs*maxlogθ0
	niter = 10000
    #f, fn = fMALA, "MALA" 
	f, fn = fRWMH, "RWMH"
	pal = palette(:tab10)
    fs = 12
	plotoptions = Dict(
					:xticks => [(10.)^i for i in -7:7 if iseven(i)],
					:xscale => :log10,
					:minorticks => 1,
					:linewidth => 2,
					:guidefontsize => fs, 
					:tickfontsize => fs, 
					:legendfontsize => fs,
					:dpi => 600
				)
	yticks = (10.).^(-10:0)
	pacc = plot()
	pcost = plot()
	pess = plot()
	pksess = plot()
	pjump = plot()
	pejump = plot()
	pjit = plot()
	donelegend = false

	targets = [(Normal(), "Normal", :solid), (Laplace(), "Laplace", :dash), (Cauchy(), "Cauchy", :dot)]
	for (target, tn, ls) in targets
        auto_log_acceptance_ratio, auto_jump, auto_ess, auto_KSess, auto_cost, auto_ejump, auto_jit,
		   jitauto_log_acceptance_ratio, jitauto_jump, jitauto_ess, jitauto_KSess, jitauto_cost, jitauto_ejump, jitauto_jit = run(f, target, logθ0s, niter)

		if !donelegend
			autolbl = "AutoStep $fn"
			jitautolbl = "Jittered AutoStep $fn"
			donelegend = true
		else
			autolbl = ""
			jitautolbl = ""
		end

		plot!(pacc, (10.).^logθ0s, exp.(auto_log_acceptance_ratio), yscale=:log10, ylimits=((10.)^(-2), 1.2), yticks=yticks, label=autolbl, color=pal[1], linestyle=ls; plotoptions...)
		plot!(pacc, (10.).^logθ0s, exp.(jitauto_log_acceptance_ratio), yscale=:log10, ylimits = ((10.)^(-2), 1.2), yticks=yticks, legend=:bottomleft, label=jitautolbl, color=pal[2], linestyle=ls; plotoptions...)
    	plot!(pcost, (10.).^logθ0s, auto_cost/niter, label=autolbl, color=pal[1], linestyle=ls; plotoptions...)
		plot!(pcost, (10.).^logθ0s, jitauto_cost/niter, label=jitautolbl, color=pal[2], legend=:top, linestyle=ls; plotoptions...)
		plot!(pess, (10.).^logθ0s, auto_ess ./ auto_cost, yscale=:log10, label=autolbl, color=pal[1], linestyle=ls; plotoptions...)
		plot!(pess, (10.).^logθ0s, jitauto_ess ./ jitauto_cost, yscale=:log10, label=jitautolbl, color=pal[2], linestyle=ls; plotoptions...)
    	plot!(pksess, (10.).^logθ0s, auto_KSess ./ auto_cost, yscale=:log10, yticks=yticks, label=autolbl, color=pal[1], linestyle=ls; plotoptions...)
		plot!(pksess, (10.).^logθ0s, jitauto_KSess ./ jitauto_cost, yscale=:log10, yticks=yticks, label=jitautolbl, color=pal[2], linestyle=ls; plotoptions...)
		plot!(pjump, (10.).^logθ0s, auto_jump ./ (auto_cost/niter), yscale=:log10, yticks=yticks, ylimits=(1e-6,1), label=autolbl, color=pal[1], legend=:bottom, linestyle=ls; plotoptions...)
		plot!(pjump, (10.).^logθ0s, jitauto_jump ./ (jitauto_cost/niter), yscale=:log10, yticks=yticks, ylimits=(1e-6, 1), label=jitautolbl, color=pal[2], linestyle=ls, legend=:bottomleft; plotoptions...)
        plot!(pejump, (10.).^logθ0s, auto_ejump, yscale=:log10, yticks=yticks, ylimits=(1e-6,1), label=autolbl, color=pal[1], legend=:bottom, linestyle=ls; plotoptions...)
		plot!(pejump, (10.).^logθ0s, jitauto_ejump, yscale=:log10, yticks=yticks, ylimits=(1e-6, 1), label=jitautolbl, color=pal[2], linestyle=ls, legend=:bottomleft; plotoptions...)
        plot!(pjit, (10.).^logθ0s, auto_jit, yscale=:log10, label=autolbl, color=pal[1], ylimits=((10.)^(-2), 1.2), yticks=yticks, linestyle=ls; plotoptions...)
		plot!(pjit, (10.).^logθ0s, jitauto_jit, yscale=:log10, label=jitautolbl, color=pal[2], ylimits=((10.)^(-2), 1.2), yticks=yticks, linestyle=ls; plotoptions...)

		#hack to get line style legend
		plot!(pacc, [((10.).^logθ0s)[1]], [(exp.(auto_log_acceptance_ratio))[1]], color=:black, linestyle=ls, label=tn, dpi=600)
		plot!(pcost, [((10.).^logθ0s)[1]], [(auto_cost/niter)[1]], color=:black, linestyle=ls, label=tn, dpi=600)
		plot!(pess, [((10.).^logθ0s)[1]], [(auto_ess ./ auto_cost)[1]], color=:black, linestyle=ls, label=tn, dpi=600)
		plot!(pksess, [((10.).^logθ0s)[1]], [(auto_KSess ./ auto_cost)[1]], color=:black, linestyle=ls, label=tn, dpi=600)
		plot!(pjump, [((10.).^logθ0s)[1]], [(auto_jump ./ (auto_cost/niter))[1]], color=:black, linestyle=ls, label=tn, dpi=600)
		plot!(pejump, [((10.).^logθ0s)[1]], [auto_jump[1]], color=:black, linestyle=ls, label=tn, dpi=600)
		plot!(pjit, [((10.).^logθ0s)[1]], [auto_jit[1]], color=:black, linestyle=ls, label=tn, dpi=600)

	end
	xlabel!(pacc, "\$\\theta_0\$")
	ylabel!(pacc, "Mean Acceptance Prob")
	xlabel!(pcost, "\$\\theta_0\$")
	ylabel!(pcost, "Mean Involutions per Iter")
	xlabel!(pess, "\$\\theta_0\$")
	ylabel!(pess, "ESS per Cost")
	xlabel!(pksess, "\$\\theta_0\$")
	ylabel!(pksess, "KSESS per Cost")
	xlabel!(pjump, "\$\\theta_0\$")
	ylabel!(pjump, "Mean Jump Dist per Cost")
    xlabel!(pejump, "\$\\theta_0\$")
	ylabel!(pejump, "Mean Energy Jump Dist per Iter")
    xlabel!(pjit, "\$\\theta_0\$")
	ylabel!(pjit, "Mean Jitter Prob")
	savefig(pacc, "jit_acc_$fn.png")
	savefig(pjump, "jit_jump_$fn.png")
	savefig(pejump, "jit_ejump_$fn.png")
	savefig(pksess, "jit_ksess_$fn.png")
	savefig(pess, "jit_ess_$fn.png")
	savefig(pcost, "jit_cost_$fn.png")
	savefig(pjit, "jit_jit_$fn.png")
end

main()
