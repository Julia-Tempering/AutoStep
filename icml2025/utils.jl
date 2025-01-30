using Statistics, StatsBase, LogDensityProblems, DataFrames, CSV, ForwardDiff
using Suppressor, BridgeStan, HypothesisTests, Pigeons, JSON, LinearAlgebra

# define orbital model 
include("orbital_model_definition.jl")
orbital_model = Octofitter.LogDensityModel(GL229A; autodiff = :ForwardDiff, verbosity = 4)
""" 
Notes for Ivy: 

Octofitter's LogDensityModel already also conforms to the LogDensityProblems interface.
However, please not that you may need to post-transform samples to get them to 
look like on the Pigeons GitHub issue link.  
--- See `model.link` or `model.invlink`. in the Octofitter documentation for more information
E.g., William has already defined for us: 
	LogDensityProblems.logdensity(p::LogDensityModel, θ) = p.ℓπcallback(θ)
	LogDensityProblems.logdensity_and_gradient(p::LogDensityModel, θ) = p.∇ℓπcallback(θ)
	LogDensityProblems.dimension(p::LogDensityModel{D}) where D = D
	LogDensityProblems.capabilities(::Type{<:LogDensityModel}) = LogDensityProblems.LogDensityOrder{1}()
"""

# convert model to pigeon-digestable model
function model_to_target(model)
	if startswith(model, "funnel2")
		return Pigeons.stan_funnel(1, 0.6)
	elseif startswith(model, "funnel100")
		return Pigeons.stan_funnel(99, 6.0)
	elseif startswith(model, "mRNA")
		stan_example_path(name) = dirname(dirname(pathof(Pigeons))) * "/examples/$name"
		return StanLogPotential(stan_example_path("stan/mRNA.stan"), "icml2025/data/mRNA.json")
	elseif startswith(model, "sonar")
		return StanLogPotential("icml2025/data/horseshoe_logit.stan", "icml2025/data/sonar.json")
	elseif startswith(model, "ionosphere")
		return StanLogPotential("icml2025/data/horseshoe_logit.stan", "icml2025/data/ionosphere.json")
	elseif startswith(model, "prostate")
		return StanLogPotential("icml2025/data/horseshoe_logit.stan", "icml2025/data/prostate.json")
	elseif startswith(model, "kilpisjarvi")
		include(joinpath(dirname(dirname(pathof(Pigeons))), "test", "supporting", "postdb.jl"))
		return log_potential_from_posterior_db("kilpisjarvi_mod-kilpisjarvi.json")
	elseif startswith(model, "orbital")
		return orbital_model
	else
		error("unknown model $model")
	end
end

# Define the Funnel model.
struct Funnel
	dim::Int
	scale::Float64
end
function LogDensityProblems.logdensity(model::Funnel, x)
	try
        return logpdf(Normal(0, 3), x[1]) + sum(logpdf.(Normal(0, exp(x[1] / model.scale)), x[2:end]))
    catch e 
        return -Inf
    end
end
LogDensityProblems.dimension(model::Funnel) = model.dim + 1
LogDensityProblems.capabilities(::Funnel) = LogDensityProblems.LogDensityOrder{0}()

# Define the kilpisjarvi model
struct Kilpisjarvi
	x::Array
	y::Array
	N::Int
	xpred::Float64
	pmualpha::Float64
	psalpha::Float64
	pmubeta::Float64
	psbeta::Float64
end
function LogDensityProblems.logdensity(model::Kilpisjarvi, x)
	alpha, beta, sigma = x
	if sigma <= 0
		return -Inf
	end
	# priors
	log_prior_alpha = logpdf(Normal(model.pmualpha, model.psalpha), alpha)
	log_prior_beta = logpdf(Normal(model.pmubeta, model.psbeta), beta)
	log_prior_sigma = logpdf(Truncated(Normal(0, 1), 0, Inf), sigma)
	# logpdf(Normal(0, 1), sigma) - log(ccdf(Normal(0, 1), 0))  # Adjust for truncation
	# log likelihood
	log_likelihood = sum(logpdf(Normal(alpha + beta * model.x[i], sigma), model.y[i]) for i in 1:model.N)

	return log_prior_alpha + log_prior_beta + log_prior_sigma + log_likelihood
end
LogDensityProblems.dimension(model::Kilpisjarvi) = 3
LogDensityProblems.capabilities(::Kilpisjarvi) = LogDensityProblems.LogDensityOrder{0}()

# Define the mRNA model
struct mRNA
	N::Int
	ts::Vector{Float64}
	ys::Vector{Float64}
end
function exp_a_minus_exp_b(a, b)
	return a > b ? -exp(a) * expm1(b - a) : exp(b) * expm1(a - b)
end
function get_mu(tmt0, km0, beta, delta)
	if tmt0 <= 0.0
		return 0.0  # must force mu=0 when t < t0 (reaction hasn't started)
	end
	dmb = delta - beta
	if abs(dmb) < eps()  # `eps()` gives machine precision in Julia
		return km0 * tmt0
	else
		return km0 * exp_a_minus_exp_b(-beta * tmt0, -delta * tmt0) / dmb
	end
end
function LogDensityProblems.logdensity(model::mRNA, x)
	lt0, lkm0, lbeta, ldelta, lsigma = x
	# transform
	t0 = 10.0^lt0
	km0 = 10.0^lkm0
	beta = 10.0^lbeta
	delta = 10.0^ldelta
	sigma = 10.0^lsigma
	# prior
	log_prior_lt0 = logpdf(Uniform(-2, 1), lt0)
	log_prior_lkm0 = logpdf(Uniform(-5, 5), lkm0)
	log_prior_lbeta = logpdf(Uniform(-5, 5), lbeta)
	log_prior_ldelta = logpdf(Uniform(-5, 5), ldelta)
	log_prior_lsigma = logpdf(Uniform(-2, 2), lsigma)
	# log likelihood
	log_likelihood = sum(logpdf(Normal(get_mu(model.ts[i] - t0, km0, beta, delta), sigma), model.ys[i]) for i in 1:model.N)
	return log_prior_lt0 + log_prior_lkm0 + log_prior_lbeta + log_prior_ldelta + log_prior_lsigma + log_likelihood
end
LogDensityProblems.dimension(model::mRNA) = 5
LogDensityProblems.capabilities(::mRNA) = LogDensityProblems.LogDensityOrder{0}()

# Define the horseshoe models
struct Horseshoe
	x::Array
	y::Array
	n::Int
	d::Int
end
function LogDensityProblems.logdensity(model::Horseshoe, x)
	tau = x[1]  # Global shrinkage parameter
	lambda = x[2:model.d+1]  # Local shrinkage parameters (vector of length d)
	beta0 = x[model.d+2]  # Intercept
	beta = x[model.d+3:end]  # Regression coefficients (vector of length d)
	# priors
	log_prior_tau = logpdf(TDist(1), tau)  # Cauchy(0,1) prior for tau
	log_prior_lambda = sum(logpdf(TDist(1), λ) for λ in lambda)  # Cauchy(0,1) priors for lambda
	log_prior_beta0 = logpdf(LocationScale(0, 1, TDist(3)), beta0)  # Student-T(3) prior for intercept
	log_prior_beta = logpdf(MvNormal(zeros(model.d), (tau .* lambda)), beta)  # Horseshoe prior for beta
	# log likelihood (Bernoulli logit model)
	log_likelihood = sum(logpdf(BernoulliLogit(beta0 + dot(model.x[i, :], beta)), model.y[i]) for i in 1:model.n)

	return log_prior_tau + log_prior_lambda + log_prior_beta0 + log_prior_beta + log_likelihood
end
LogDensityProblems.dimension(model::Horseshoe) = 2 * model.d + 2
LogDensityProblems.capabilities(::Horseshoe) = LogDensityProblems.LogDensityOrder{0}()

# Define the function for log density and its gradient
function LogDensityProblems.logdensity_and_gradient(model::Union{Funnel, mRNA, Kilpisjarvi, Horseshoe}, x)
	logp = LogDensityProblems.logdensity(model, x)
	grad_logp = ForwardDiff.gradient(z -> LogDensityProblems.logdensity(model, z), x)
	return logp, grad_logp
end

# utility function to match models for all kernels
function logdens_model(model, data)
	if startswith(model, "funnel")
		return Funnel(data["dim"], data["scale"])
	elseif startswith(model, "kilpisjarvi")
		return Kilpisjarvi(data["x"], data["y"], data["N"], data["xpred"], data["pmualpha"],
			data["psalpha"], data["pmubeta"], data["psbeta"])
	elseif startswith(model, "mRNA")
		return mRNA(data["N"], data["ts"], data["ys"])
	elseif startswith(model, "sonar") || startswith(model, "prostate") || startswith(model, "ionosphere")
		return Horseshoe(hcat(data["x"]...)', data["y"], data["n"], data["d"])
	elseif startswith(model, "orbital")
		return orbital_model
	else
		error("unknown model $model")
	end
end

function stan_data(model::String; dataset = nothing, dim = nothing, scale = nothing)
	if startswith(model, "funnel2")
		Dict("dim" => 1, "scale" => 0.3)
	elseif startswith(model, "funnel100")
		Dict("dim" => 99, "scale" => 6.0)
	elseif startswith(model, "orbital")
		nothing
	else
		file_name = if startswith(model, "kilpisjarvi")
			"kilpisjarvi_mod"
		elseif startswith(model, "sonar")
			"sonar"
		elseif startswith(model, "prostate")
			"prostate"
		elseif startswith(model, "ionosphere")
			"ionosphere"
		elseif startswith(model, "mRNA")
			"mRNA"
		else
			error("unknown model $model")
		end
		JSON.parse(read(joinpath("icml2025", "data", file_name * ".json"), String))
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
get_component_samples(samples::DataFrame, idx_component) = Array(samples[:, idx_component])

# returns tuple: (margin_idx, mean, std_dev)
special_margin_mean_std(model::String, args...) =
	if startswith(model, "funnel")
		(1, 0.0, 3.0)
	elseif startswith(model, "banana")
		(1, 0.0, sqrt(10))
	elseif startswith(model, "eight_schools")    # approx using variational PT (see 8schools_pilot.jl)
		(1, 3.574118538746056, 3.1726880307401455)
	elseif startswith(model, "normal")
		(1, 0.0, 1.0)
	elseif startswith(model, "two_component_normal")
		(1, 0.0, first(two_component_normal_stdevs(args...))) # the margin with the largest stdev
	elseif startswith(model, "eight_schools_noncentered")
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
n_vars(samples::DataFrame) = size(samples, 2)

to_chains(samples::AbstractVector) = Chains(samples)
to_chains(samples::DataFrame) = Chains(Array(samples))

function df_to_vec(df::DataFrame)
	n = size(df, 1)
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
	B = Int(ceil(T / N))

	# check for bad failure first
	res = @suppress ExactOneSampleKSTest(xs, target)
	ess2 = (log(2) * sqrt(π / 2) / res.δ)^2
	if ess2 ≤ T * (log(2) * sqrt(π / 2))^2 / B
		return ess2
	end

	# reasonably functioning; get better ess estimate in this regime
	batches = [i:min(T, i + B - 1) for i in 1:B:T]
	if length(batches[end]) < B
		b = pop!(batches)
		batches[end] = (batches[end][begin]):(b[end])
	end
	s = 0
	for b in batches
		res = @suppress ExactOneSampleKSTest(xs[b], target)
		s += sqrt(length(b)) * res.δ
	end
	s /= (length(batches) * log(2) * sqrt(π / 2))
	ess2 = T * s^(-2)
	return ess2
end


# with unknown distribution
function twosample_sortedref(xs, ysort)
	ny = length(ysort)
	nx = length(xs)
	δx = 1.0 / nx
	δy = 1.0 / ny
	xsort = sort(xs)
	curFy = 0.0
	curiy = 0
	curFx = 0.0
	curix = 0
	δ = 0.0
	while (curiy < ny) || (curix < nx)
		if (curiy == ny)
			curix += 1
			curFx += δx
		elseif (curix == nx)
			curiy += 1
			curFy += δy
		elseif (xsort[curix+1] < ysort[curiy+1])
			curix += 1
			curFx += δx
		else
			curiy += 1
			curFy += δy
		end
		δ = max(δ, abs(curFx - curFy))
	end
	return δ
end

function KSess_two_sample(xs, ys)
	T = length(xs)
	N = 40
	B = Int(ceil(T / N))

	ysort = sort(ys)

	# check for bad failure first
	δ = twosample_sortedref(xs, ysort) #@suppress ApproximateTwoSampleKSTest(xs, ys)
	ess2 = (log(2) * sqrt(π / 2) / δ)^2
	if ess2 ≤ T * (log(2) * sqrt(π / 2))^2 / B
		return ess2
	end

	# reasonably functioning; get better ess estimate in this regime
	batches = [i:min(T, i + B - 1) for i in 1:B:T]
	if length(batches[end]) < B
		b = pop!(batches)
		batches[end] = (batches[end][begin]):(b[end])
	end
	s = 0.0
	for b in batches
		δ = twosample_sortedref(xs[b], ysort) #@suppress ApproximateTwoSampleKSTest(xs[b], ys)
		s += sqrt(length(b)) * δ
	end
	s /= (length(batches) * log(2) * sqrt(π / 2))
	ess2 = T * s^(-2)
	return ess2
end

# get the minimum ksess of all margins
function min_KSess(samples, model)
	if startswith(model, "funnel100")
		reference_samples = CSV.read("icml2025/samples/funnel2.csv", DataFrame)
		miness = KSess_two_sample(getindex.(samples, 1), collect(reference_samples[1, :]))
		#ys = collect(reference_samples[2,:])
		#for i in 2:length(samples[1])
		#    miness = min(miness, KSess_two_sample(getindex.(samples, i), ys))
		#end
		return miness
	else
		miness = Inf
		try
			reference_samples = CSV.read("icml2025/samples/$(model).csv", DataFrame)
			for i in 1:length(samples[1])
				ys = collect(reference_samples[i, :])
				miness = min(miness, KSess_two_sample(getindex.(samples, i), ys))
			end
			return miness
		catch e
			return 0.0
		end
	end
end
