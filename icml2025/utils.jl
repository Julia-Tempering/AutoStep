using Statistics, StatsBase, LogDensityProblems, DataFrames, CSV
using Suppressor

# Define the Funnel model.
struct Funnel
    dim::Int
    scale::Float64
end
function LogDensityProblems.logdensity(model::Funnel, x)
    dim = LogDensityProblems.dimension(model)
    # Ensure x has the correct dimensions.
    if length(x) != dim
        return -Inf
    end
    return logpdf(Normal(0, 3), x[1]) + sum(logpdf.(Normal(0, exp(x[1] / model.scale)), x[2:end]))
end
LogDensityProblems.dimension(model::Funnel) = model.dim + 1
LogDensityProblems.capabilities(::Funnel) = LogDensityProblems.LogDensityOrder{0}()


# utility function to match models for all kernels
function logdens_model(model, data)
	if startswith(model, "funnel")
		return Funnel(data["dim"], data["scale"])
	else
		error("unknown model $model")
	end
end

function stan_data(model::String; dataset = nothing, dim = nothing, scale = nothing)
	if startswith(model, "funnel2")
		Dict("dim" => 1.0, "scale" => 0.1)
	elseif startswith(model, "funnel100")
		Dict("dim" => 99, "scale" => 0.03)
	else
		file_name = if startswith(model, "kilpisjarvi")
			"kilpisjarvi_mod"
		elseif startswith(model, "sonar")
			"sonar"
		elseif startswith(model, "mRNA")
			"mRNA"
		else
			error("unknown model $model")
		end
		JSON.parse(read(joinpath("data", file_name * ".json"), String))
	end
end


###############################################################################
# ESS and friends
###############################################################################

function margin_ess(samples, model, args...)
	margin_idx, true_mean, true_sd = special_margin_mean_std(model, args...)
	margin_samples = get_component_samples(samples, margin_idx)
	batch_means_ess(margin_samples, true_mean, true_sd)
end

# minimum ess using batch means
min_ess_batch(samples) =
	minimum(1:n_vars(samples)) do i
		batch_means_ess(get_component_samples(samples, i))
	end

# minimum ess using the MCMCChains approach (which is based on Stan's)
function min_ess_chains(samples)
	chn = to_chains(samples)
	min(
		minimum(ess(chn).nt.ess),            # default is :bulk which computes ess on rank-normalized vars
		minimum(ess(chn, kind = :basic).nt.ess), # :basic is actual ess of the vars.
	)
end

# minimum over dimensions and methods
function min_ess_all_methods(samples, model)
	special_margin_ess = try
		margin_ess(samples, model)
	catch e
		if e isa KeyError
			Inf
		else
			rethrow(e)
		end
	end
	min(special_margin_ess, min(min_ess_chains(samples), min_ess_batch(samples)))
end

# returns ESS, mean and var
function margin_summary(samples, model, args...)
	margin_idx, true_mean, true_sd = special_margin_mean_std(model, args...)
	margin_samples = get_component_samples(samples, margin_idx)
	margin_ess_exact = batch_means_ess(margin_samples, true_mean, true_sd)
	margin_mean = mean(margin_samples)
	margin_var = var(margin_samples)
	DataFrame(margin_ess_exact = margin_ess_exact, margin_mean = margin_mean, margin_var = margin_var)
end


batch_means_ess(samples::AbstractVector) =
	batch_means_ess(samples, mean_and_std(samples)...)
function batch_means_ess(
	samples::AbstractVector,
	posterior_mean::Real,
	posterior_sd::Real,
)
	n_samples = length(samples)
	n_blocks = 1 + isqrt(n_samples)
	blk_size = n_samples ÷ n_blocks # takes floor of division
	centered_batch_means = map(1:n_blocks) do b
		i_start = blk_size * (b - 1) + 1
		i_end   = blk_size * b
		mean(x -> (x - posterior_mean) / posterior_sd, @view samples[i_start:i_end])
	end
	n_blocks / mean(abs2, centered_batch_means)
end




###############################################################################
# utils for processing samples
###############################################################################

get_component_samples(samples::AbstractVector, idx_component::Int) =
    [s[idx_component] for s in samples]
get_component_samples(samples::DataFrame, idx_component) = Array(samples[:,idx_component])

# returns tuple: (margin_idx, mean, std_dev)
special_margin_mean_std(model::String, args...) = 
    if startswith(model,"funnel")
        (1, 0., 3.)
    elseif startswith(model,"banana")
        (1, 0., sqrt(10))
    elseif startswith(model,"eight_schools")    # approx using variational PT (see 8schools_pilot.jl)
        (1, 3.574118538746056, 3.1726880307401455)
    elseif startswith(model, "normal") 
        (1, 0., 1.)
    elseif startswith(model, "two_component_normal")
        (1, 0., first(two_component_normal_stdevs(args...)) ) # the margin with the largest stdev
    elseif startswith(model,"eight_schools_noncentered")
        (1, 0.316857, 0.988146)
    elseif startswith(model, "garch11")
        (1, 5.05101, 0.12328)
    elseif startswith(model, "gp_pois_regr") 
        (1, 5.68368, 0.708641)
    elseif startswith(model, "lotka_volterra")
        (1, 0.547547, 0.0636274)
    elseif startswith(model, "kilpisjarvi") 
        (1, -58.2412, 30.3941)
    elseif startswith(model, "logearn_logheight_male")
        (1, 3.64354, 2.71194)
    elseif startswith(model, "diamonds")
        (1, 6.72838, 0.22442)
    elseif model == "mRNA"
        (3, -1.8909, 1.0014) # 3 => log(beta), hardest to sample together with delta
    elseif startswith(model, "horseshoe")
        (1, 0.88484, 0.116685)
    else
        throw(KeyError(model))
    end
n_vars(samples::AbstractVector) = length(first(samples))
n_vars(samples::DataFrame) = size(samples,2)

to_chains(samples::AbstractVector) = Chains(samples)
to_chains(samples::DataFrame) = Chains(Array(samples))

function df_to_vec(df::DataFrame) 
    n = size(df,1)
    df_vec = Vector{Vector{Float64}}(undef, n) 
    for i in 1:n 
        df_vec[i] = Vector(df[i, :]) 
    end 
    return df_vec
end 



###############################################################################
# utils for ksess metrics
###############################################################################

# with known distribution
function KSess_one_sample(xs, target)
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


# with unknown distribution
function KSess_two_sample(xs, target)
	T = length(xs)
	N = 40
	B = Int(ceil(T/N))

	# check for bad failure first
	res = @suppress ApproximateTwoSampleKSTest(xs, target)
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
		res = @suppress ApproximateTwoSampleKSTest(xs[b], target)
		s += sqrt(length(b))*res.δ
	end
	s /= (length(batches) * log(2) * sqrt(π/2))
	ess2 = T*s^(-2)
	return T*s^(-2)
end

# get the minimum ksess of all margins
function min_KSess(samples, model)
    if startswith(model, "funnel")
        target1 = Normal(0, 3)
        miness = KSess_one_sample(getindex.(samples, 1), target1)
        target2 = CSV.read("icml2025/samples/funnel2.csv", DataFrame)
        target2 = collect(target2[2,:])
        for i in 2:length(samples[1])
            miness = min(miness, KSess_two_sample(getindex.(samples, i), target2))
        end
        return miness
    end
end