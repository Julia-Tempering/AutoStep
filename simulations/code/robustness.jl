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
	T = length(xs)
	N = 40
	B = Int(ceil(T/N))

	# check for bad failure first
	res = @suppress ExactOneSampleKSTest(xs, target)
	ess2 = (log(2) * sqrt(π/2) / res.δ)^2
	if ess2 ≤ T*(log(2) * sqrt(π/2))^2/B
		return ess2
	end
	
	# reasonably functioning; get better ess estimate in this regime
	batches = [i:min(T, i+B-1) for i in 1:B:T]
	if length(batches[end]) < B
		b = pop!(batches)
		batches[end] = (batches[end][begin]):(b[end])
	end
	s = 0
	for b in batches
		res = @suppress ExactOneSampleKSTest(xs[b], target)
		s += sqrt(length(b))*res.δ
	end
	s /= (length(batches) * log(2) * sqrt(π/2))
	ess2 = T*s^(-2)
	return T*s^(-2)
end

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

function run(f, target, logθ0s, niter)
	auxtarget = Normal(0., 1.)
	initial = target	
	auto_log_acceptance_ratio = []
	auto_jump = []
	auto_ejump = []
	auto_ess = []
	auto_KSess = []
	auto_cost = []
    fix_log_acceptance_ratio = []
    fix_jump = []
    fix_ejump = []
	fix_ess = []
	fix_KSess = []
	fix_cost = []
	for logθ0 in logθ0s
		θ0 = (10.)^logθ0
		# autostep
		x0 = rand(target)
		xs, cs, logas, ejumps, jitters, θs = mcmc(x0, auto_step, f, θ0, target, auxtarget, niter)
		maxlogas = maximum(logas)
		logas .-= maxlogas
		push!(auto_log_acceptance_ratio, maxlogas + log(mean(exp.(logas))))
		push!(auto_jump, mean(abs.(xs[2:end] .- xs[1:end-1])))
		push!(auto_ess, ess(xs, target))
		push!(auto_KSess, KSess(xs, target))
		push!(auto_cost, cs[end])
		push!(auto_ejump, mean(abs.(ejumps)))
			
		# fixstep
        xs, cs, logas, ejumps, jitters, θs = mcmc(x0, fix_step, f, θ0, target, auxtarget, niter)
        maxlogas = maximum(logas)
		logas .-= maxlogas
		push!(fix_log_acceptance_ratio, maxlogas + log(mean(exp.(logas))))
		push!(fix_jump, mean(abs.(xs[2:end] .- xs[1:end-1])))
		push!(fix_ess, ess(xs, target))
		push!(fix_KSess, KSess(xs, target))
		push!(fix_cost, cs[end])
		push!(fix_ejump, mean(abs.(ejumps)))
	end	
	return auto_log_acceptance_ratio, auto_jump, auto_ess, auto_KSess, auto_cost, auto_ejump,
			fix_log_acceptance_ratio, fix_jump, fix_ess, fix_KSess, fix_cost, fix_ejump
end


function main()
	minlogθ0 = -7
	maxlogθ0 = 7
	Nθs = 100
	logθ0s = (Nθs .- Array(0:Nθs))/Nθs*minlogθ0 + Array(0:Nθs)/Nθs*maxlogθ0
	niter = 1000000
    f, fn = fMALA, "MALA" 
	#f, fn = fRWMH, "RWMH"
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
	donelegend = false

	targets = [(Normal(), "Normal", :solid), (Laplace(), "Laplace", :dash), (Cauchy(), "Cauchy", :dot)]
	for (target, tn, ls) in targets
        auto_log_acceptance_ratio, auto_jump, auto_ess, auto_KSess, auto_cost, auto_ejump,
		   fix_log_acceptance_ratio, fix_jump, fix_ess, fix_KSess, fix_cost, fix_ejump = run(f, target, logθ0s, niter)

		if !donelegend
			autolbl = "AutoStep $fn"
			fixlbl = "$fn"
			donelegend = true
		else
			autolbl = ""
			fixlbl = ""
		end

		plot!(pacc, (10.).^logθ0s, exp.(auto_log_acceptance_ratio), yscale=:log10, ylimits=((10.)^(-2), 1.2), yticks=yticks, label=autolbl, color=pal[1], linestyle=ls; plotoptions...)
		plot!(pacc, (10.).^logθ0s, exp.(fix_log_acceptance_ratio), yscale=:log10, ylimits = ((10.)^(-2), 1.2), yticks=yticks, legend=:bottomleft, label=fixlbl, color=pal[2], linestyle=ls; plotoptions...)
    	plot!(pcost, (10.).^logθ0s, auto_cost/niter, label=autolbl, color=pal[1], linestyle=ls; plotoptions...)
		plot!(pcost, (10.).^logθ0s, fix_cost/niter, label=fixlbl, color=pal[2], legend=:top, linestyle=ls; plotoptions...)
		plot!(pess, (10.).^logθ0s, auto_ess ./ auto_cost, yscale=:log10, label=autolbl, color=pal[1], linestyle=ls; plotoptions...)
		plot!(pess, (10.).^logθ0s, fix_ess ./ fix_cost, yscale=:log10, label=fixlbl, color=pal[2], linestyle=ls; plotoptions...)
    	plot!(pksess, (10.).^logθ0s, auto_KSess ./ auto_cost, yscale=:log10, yticks=yticks, label=autolbl, color=pal[1], linestyle=ls; plotoptions...)
		plot!(pksess, (10.).^logθ0s, fix_KSess ./ fix_cost, yscale=:log10, yticks=yticks, label=fixlbl, color=pal[2], linestyle=ls; plotoptions...)
		plot!(pjump, (10.).^logθ0s, auto_jump ./ (auto_cost/niter), yscale=:log10, yticks=yticks, ylimits=(1e-6,1), label=autolbl, color=pal[1], legend=:bottom, linestyle=ls; plotoptions...)
		plot!(pjump, (10.).^logθ0s, fix_jump ./ (fix_cost/niter), yscale=:log10, yticks=yticks, ylimits=(1e-6, 1), label=fixlbl, color=pal[2], linestyle=ls, legend=:bottomleft; plotoptions...)
        plot!(pejump, (10.).^logθ0s, auto_ejump, yscale=:log10, yticks=yticks, ylimits=(1e-6,1), label=autolbl, color=pal[1], legend=:bottom, linestyle=ls; plotoptions...)
		plot!(pejump, (10.).^logθ0s, fix_ejump, yscale=:log10, yticks=yticks, ylimits=(1e-6, 1), label=fixlbl, color=pal[2], linestyle=ls, legend=:bottomleft; plotoptions...)

		#hack to get line style legend
		plot!(pacc, [((10.).^logθ0s)[1]], [(exp.(auto_log_acceptance_ratio))[1]], color=:black, linestyle=ls, label=tn, dpi=600)
		plot!(pcost, [((10.).^logθ0s)[1]], [(auto_cost/niter)[1]], color=:black, linestyle=ls, label=tn, dpi=600)
		plot!(pess, [((10.).^logθ0s)[1]], [(auto_ess ./ auto_cost)[1]], color=:black, linestyle=ls, label=tn, dpi=600)
		plot!(pksess, [((10.).^logθ0s)[1]], [(auto_KSess ./ auto_cost)[1]], color=:black, linestyle=ls, label=tn, dpi=600)
		plot!(pjump, [((10.).^logθ0s)[1]], [(auto_jump ./ (auto_cost/niter))[1]], color=:black, linestyle=ls, label=tn, dpi=600)
		plot!(pejump, [((10.).^logθ0s)[1]], [auto_jump[1]], color=:black, linestyle=ls, label=tn, dpi=600)

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
	savefig(pacc, "acc_$fn.png")
	savefig(pjump, "jump_$fn.png")
	savefig(pejump, "ejump_$fn.png")
	savefig(pksess, "ksess_$fn.png")
	savefig(pess, "ess_$fn.png")
	savefig(pcost, "cost_$fn.png")
end

main()
