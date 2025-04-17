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

function run(f, target, initial_logθ0s, rounds)
	auxtarget = Normal(0., 1.)
	logθ0_traces = []
	cost_traces = []
	for logθ0 in initial_logθ0s
		θ0 = (10.)^logθ0
		push!(logθ0_traces, [])
		push!(cost_traces, [])
		x0 = rand(Normal(0., 20.))
		for r = 1:rounds
			println("Initial Logθ0 $logθ0 Target $target Round $r Kernel $f")
			push!(logθ0_traces[end], log10(θ0))
			xs, cs, logas, ejumps, jitters, θs = mcmc(x0, auto_step, f, θ0, target, auxtarget, 2^r)
			push!(cost_traces[end], cs[end]/2^r)
			θ0 = θ0*2^median(log2.(θs) .- log2(θ0))
			x0 = copy(xs[end])
		end
	end	
	return logθ0_traces, cost_traces
end


function main()
	initial_logθ0s = -7:7
	rounds = 20
	pal = palette(:tab10)
    fs = 12
	plotoptions = Dict(
					:yticks => [(10.).^i for i in -7:7 if iseven(i)],
					:minorticks => 1,
					:linewidth => 1,
					:guidefontsize => fs, 
					:tickfontsize => fs, 
					:legendfontsize => fs,
					:dpi => 600
				)
        plotoptionsc = Dict(
					:linewidth => 1,
					:guidefontsize => fs, 
					:tickfontsize => fs, 
					:legendfontsize => fs,
					:dpi => 600
				)
	p = plot()
	pc = plot()

	targets = [(Normal(), "Normal", :solid), (Laplace(), "Laplace", :dash), (Cauchy(), "Cauchy", :dot)]
	rwmh_legend = false
	crwmh_legend = false
	mala_legend = false
	cmala_legend = false
	for (target, tn, ls) in targets
        	logθ0_traces_RWMH, cost_traces_RWMH = run(fRWMH, target, initial_logθ0s, rounds)
        	logθ0_traces_MALA, cost_traces_MALA = run(fMALA, target, initial_logθ0s, rounds)

		for logθ0_trace in logθ0_traces_RWMH
			lbl = (!rwmh_legend) ? "AutoStep RWMH" : ""
			rwmh_legend = true
			plot!(p, 1:rounds, (10.).^(logθ0_trace), yscale=:log10, ylimits=((10.)^(-7), (10.)^7), color=pal[1], label = lbl, linestyle=ls; plotoptions...)
		end

        	for cost_trace in cost_traces_RWMH
			lbl = (!crwmh_legend) ? "AutoStep RWMH" : ""
			crwmh_legend = true
			plot!(pc, 1:rounds, cost_trace, color=pal[1], label = lbl, linestyle=ls; plotoptionsc...)
		end
       
        	for logθ0_trace in logθ0_traces_MALA
        		lbl = (!mala_legend) ? "AutoStep MALA" : ""
			mala_legend = true
			plot!(p, 1:rounds, (10.).^(logθ0_trace), yscale=:log10, ylimits=((10.)^(-7), (10.)^7), color=pal[2], label = lbl, linestyle=ls; plotoptions...)
		end

        	for cost_trace in cost_traces_MALA
			lbl = (!cmala_legend) ? "AutoStep MALA" : ""
			cmala_legend = true
			plot!(pc, 1:rounds, cost_trace, color=pal[2], label = lbl, linestyle=ls; plotoptionsc...)
		end

		#hack to get line style legend
		plot!(p, [1], [((10.).^(logθ0_traces_RWMH[1]))[1]], yscale=:log10, ylimits=((10.)^(-7), (10.)^7), label=tn, color=:black, linestyle=ls; plotoptions...)
		plot!(pc, [1], [cost_traces_RWMH[1][1]], label=tn, color=:black, linestyle=ls; plotoptionsc...)
	end
	xlabel!(p, "Tuning Round")
	ylabel!(p, "\$\\theta_0\$")
        xlabel!(pc, "Tuning Round")
	ylabel!(pc, "Cost per Iter")
	savefig(p, "stability.png")
	savefig(pc, "stability_cost.png")
end

main()
