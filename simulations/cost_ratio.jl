include("utils.jl")

using BenchmarkTools, Zygote, ReverseDiff, ForwardDiff, Enzyme
import DifferentiationInterface as DI
import Mooncake


function grad_to_logp_ratio(model)
	pt = pigeons(
		target = model_to_target(model),
        #explorer = SliceSampler(), 
		n_rounds = 12,
        record = [traces]
	)
	my_samples = get_sample(pt)

    my_data = stan_data(model)
	my_model = logdens_model(model, my_data)
    # log_density_q(x) = LogDensityProblems.logdensity(my_model, x)
	function log_density_q(x::Vector{Float64})::Float64
		return LogDensityProblems.logdensity(my_model, x)::Float64
	end

	# Measure time for log probability
	logprob_time = @belapsed begin
		for x in my_samples
			log_density_q(x)
		end
	end

	# Measure time for gradient
	backend = ADTypes.AutoMooncake(config = nothing)
	prep = DI.prepare_gradient(log_density_q, backend, zeros(100))
	grad_time = @belapsed begin
		for x in my_samples
			# ReverseDiff.gradient(log_density_q, x)
			# enzyme
			# output = zero(x)
			# autodiff(Reverse, log_density_q, Active, Duplicated(x, output))
			# mooncake: in progress
			# 
			Mooncake.gradient(log_density_q, prep, backend, x)
		end
	end

	# Compute the ratio
	eval_ratio = grad_time / logprob_time
	println("$model: $eval_ratio")
end

for model in ["funnel100"] # "funnel2", , "kilpisjarvi", "mRNA", "horseshoe"
    grad_to_logp_ratio(model)
end


# funnel100
# ReverseDiff: 300+
# Zygote: 700+
# ForwardDiff: 70+
# Enzyme: 
# Mooncake: 

