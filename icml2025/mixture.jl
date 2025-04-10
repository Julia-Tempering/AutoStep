using CSV, DataFrames, Statistics

df = CSV.read("icml2025/exp_results_mixture_new1.csv", DataFrame)
# ratio of running time of gradient VS log potential
# computed separately, recording the avg of time_gradient/time_log_prob
function log_prob_gradient_ratio(model::AbstractString)
	if startswith(model, "sonar")
		28.69494861853798 # 28.97898926611399
	elseif startswith(model, "prostate")
		25.85595195097264 # 25.85595195097264
	elseif startswith(model, "ionosphere")
		11.211050639422963 # 11.694090350696863
	elseif startswith(model, "orbital")
		5.388536818492149 # 5.4362472566094375
	elseif startswith(model, "mRNA")
		5.766575585884267 # 5.793217015357431
	elseif startswith(model, "kilpisjarvi")
		5.956119530847897 # 6.0015265040396
	elseif startswith(model, "funnel100")
		72.55113583072277 # 71.0349931437466
	elseif startswith(model, "funnel2")
		5.960239505901212 # 5.916476226247743
    elseif startswith(model, "mixture")
        3.9580342298288507 # 3.8667905638320192
    else
		throw(KeyError(model))
	end
end
# prepare dataframe
df.cost = ifelse.(
    map(explorer -> explorer in ["AutoStep RWMH", "Adaptive RWMH", "HitAndRunSlicer"], df.explorer),
    df.n_logprob,  # non gradient-based samplers
    df.n_logprob .+ 2 * df.n_steps .* log_prob_gradient_ratio.(df.model), # 1 leapfrog = 2 gradient eval
) # gradient based: we use cost = #log_potential_eval + eta * #gradient_eval, where eta is model dependent


df.minKSess_per_cost = df.minKSess ./ df.cost
df.miness_per_cost = df.miness ./ df.cost
result = combine(groupby(df, [:explorer]),
	:miness_per_cost => median => :median_minESS_per_cost,
	:minKSess_per_cost => median => :median_minKSESS_per_cost)
print(result)

df.minKSess_per_sec = df.minKSess ./ df.time
df.miness_per_sec = df.geyerESS ./ df.time
result = combine(groupby(df, [:explorer]),
	:miness_per_sec => median => :median_minESS_per_time,
    :minKSess => median => :median_minKSess,
	:minKSess_per_sec => median => :median_minKSESS_per_time)
print(result)