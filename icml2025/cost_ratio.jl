include("utils.jl")

using Pigeons, LogDensityProblems, BenchmarkTools, ForwardDiff

function grad_to_logp_ratio(model)
	pt = pigeons(
		target = model_to_target(model),
        #explorer = SliceSampler(), 
		n_rounds = 12,
        record = [traces]
	)
	samples = get_sample(pt)

    my_data = stan_data(model)
	my_model = logdens_model(model, my_data)
    log_density_q(x) = LogDensityProblems.logdensity(my_model, x)

	# Measure time for log probability
	logprob_time = @belapsed begin
		for x in samples
			log_density_q(x)
		end
	end

	# Measure time for gradient
	grad_time = @belapsed begin
		for x in samples
			ForwardDiff.gradient(log_density_q, x)
		end
	end

	# Compute the ratio
	eval_ratio = grad_time / logprob_time
	println("$model: $eval_ratio")
end

for model in ["funnel2", "funnel100", "kilpisjarvi", "mRNA", "horseshoe"]
    grad_to_logp_ratio(model)
end
