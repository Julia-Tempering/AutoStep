using StatsPlots, CSV, DataFrames, Statistics
using CairoMakie, PairPlots

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
	else
		throw(KeyError(model))
	end
end

#=
generate pair plot, with the reference distribution
=#
function draw_pairplot(model, seed, explorer)
	df = CSV.read("icml2025/temp/$(seed)_$(model)_$(explorer).csv", DataFrame)
	df = Matrix(df)'
	df = DataFrame(df, :auto)
	df = df[:, vcat(1:5, 96:100)] # modify this for difference models
	p = pairplot(df)
	save("icml2025/plots/pairplots/pairplot_$(model)_$(explorer).png", p)
end
# for model in ["prostate"] # 
#     for explorer in ["adaptive_rwmh", "nuts", "slicer"] #"adaptive_mala", "automala", "drhmc", "autorwmh", 
#         draw_pairplot(model, 1, explorer)
#     end
# end

# helper function to create dot-line plots
function create_plot(df_filtered::DataFrame, df_label::Symbol, label::AbstractString, figure_name::AbstractString, model_mapping, explorer_mapping, legend_place::Symbol)
	df_summary = combine(groupby(df_filtered, [:model, :explorer])) do subdf
		q25, med, q75 = quantile(subdf[!, df_label], [0.25, 0.5, 0.75])
		(med = med, q25 = q25, q75 = q75)
	end
	df_summary.error_low = df_summary.med .- df_summary.q25
	df_summary.error_high = df_summary.q75 .- df_summary.med
	df_summary.model_index = getindex.(Ref(model_mapping), df_summary.model)
	df_summary.explorer_shift = [explorer_mapping[e] for e in df_summary.explorer]
	df_summary.position = df_summary.model_index .+ df_summary.explorer_shift
	p = @df df_summary StatsPlots.scatter(
		:position, :med, group = :explorer,
		yerror = (df_summary.error_low, df_summary.error_high),  # IQR range
		markersize = 8, markerstrokewidth = 1.5, marker = :circle,  # Large dots
		xlabel = "Model", ylabel = label,
		yaxis = :log10, legend = legend_place,
		yticks = 10 .^ range(floor(log10(minimum(df_summary.q25))), ceil(log10(maximum(df_summary.q75))), length = 6),  # Auto log ticks
		color = :auto,  # Ensure distinct colors per group
		margin = 3Plots.mm,
		xticks = (collect(values(model_mapping)), keys(model_mapping)),
	)
	vline!(p, [100, 300, 500, 700], linestyle = :dash, color = :grey, label = false)
	savefig("icml2025/plots/$(figure_name).png")
end

#=
comparison of all autoMCMC samplers and NUTS; experiment = "post_db"
=#
function comparison_plots(df::DataFrame)
	# prepare dataframe
	df.cost = ifelse.(
		map(explorer -> explorer in ["AutoStep RWMH", "Adaptive RWMH", "HitAndRunSlicer"], df.explorer),
		df.n_logprob,  # non gradient-based samplers
		df.n_logprob .+ 2 * df.n_steps .* log_prob_gradient_ratio.(df.model), # 1 leapfrog = 2 gradient eval
	) # gradient based: we use cost = #log_potential_eval + eta * #gradient_eval, where eta is model dependent
	df = filter(row -> !(row.explorer == "AutoStep RWMH (precond)" || row.explorer == "AutoStep MALA (precond)"), df)

	default(
		xguidefontsize = 14,  # X-axis label font size
		yguidefontsize = 14,  # Y-axis label font size
		xtickfontsize = 12,
		ytickfontsize = 12,
		legendfontsize = 12,
		size = (1000, 700),
	)
	# position helper
	model_mapping = Dict("funnel100" => 0, "funnel2" => 200, "kilpisjarvi" => 400, "mRNA" => 600, "orbital" => 800)
	explorer_mapping = Dict("AutoStep MALA" => -66, "AutoStep RWMH" => -44, "DRHMC" => -22,
		"HitAndRunSlicer" => 0, "NUTS" => 22, "adaptive MALA" => 44, "adaptive RWMH" => 66)
	sort!(df, :model) # ensure ordering on x-axis

	# minESS, minESS / sec, minESS / cost
	df_filtered = filter(row -> !isnan(row.miness), df) # need to remove NaN
	create_plot(df_filtered, :miness, "minESS", "miness", model_mapping, explorer_mapping, :bottomleft)
	df_filtered.miness_per_sec = df_filtered.miness ./ df_filtered.time
	create_plot(df_filtered, :miness_per_sec, "minESS / sec", "miness_per_sec", model_mapping, explorer_mapping, :bottomleft)
	df_filtered.miness_per_cost = df_filtered.miness ./ df_filtered.cost
	create_plot(df_filtered, :miness_per_cost, "minESS / cost", "miness_per_cost", model_mapping, explorer_mapping, :bottomleft)

	# min geyerESS
	df_filtered = filter(row -> !isnan(row.geyerESS), df) # need to remove NaN
	create_plot(df_filtered, :geyerESS, "min geyerESS", "geyerESS", model_mapping, explorer_mapping, :topright)
	df_filtered.geyerESS_per_sec = df_filtered.geyerESS ./ df_filtered.time
	create_plot(df_filtered, :geyerESS_per_sec, "min geyerESS / sec", "geyerESS_per_sec", model_mapping, explorer_mapping, :topright)
	df_filtered.geyerESS_per_cost = df_filtered.geyerESS ./ df_filtered.cost
	create_plot(df_filtered, :geyerESS_per_cost, "min geyerESS / cost", "geyerESS_per_cost", model_mapping, explorer_mapping, :topright)

	# min KSESS, min KSESS / sec, min KSESS / cost
	create_plot(df, :minKSess, "min KSESS", "minKSess", model_mapping, explorer_mapping, :topright)
	df.minKSess_per_sec = df.minKSess ./ df.time
	create_plot(df, :minKSess_per_sec, "min KSESS / sec", "minKSess_per_sec", model_mapping, explorer_mapping, :topright)
	df_filtered = filter(row -> !(row.cost == 0), df) # need to remove 0 cost
	df_filtered.minKSess_per_cost = df_filtered.minKSess ./ df_filtered.cost
	create_plot(df_filtered, :minKSess_per_cost, "min KSESS / cost", "minKSess_per_cost", model_mapping, explorer_mapping, :topright)



	# the energy jump plot
	df_filtered_no_energy = filter(row -> !(row.energy_jump_dist == 0 || isnan(row.energy_jump_dist) || !isfinite(row.energy_jump_dist)), df) # need to remove 0 cost (alg fails)
	df_summary = combine(groupby(df_filtered_no_energy, [:model, :explorer])) do subdf
		q25, med, q75 = quantile(subdf.energy_jump_dist, [0.25, 0.5, 0.75])
		(med = med, q25 = q25, q75 = q75)
	end
	df_summary.error_low = df_summary.med .- df_summary.q25
	df_summary.error_high = df_summary.q75 .- df_summary.med
	df_summary.model_index = getindex.(Ref(model_mapping), df_summary.model)
	df_summary.explorer_shift = [explorer_mapping[e] for e in df_summary.explorer]
	df_summary.position = df_summary.model_index .+ df_summary.explorer_shift
	p = @df df_summary StatsPlots.scatter(
		:position, :med, group = :explorer,
		yerror = (df_summary.error_low, df_summary.error_high),  # IQR range
		markersize = 8, markerstrokewidth = 1.5, marker = :circle,  # Large dots
		xlabel = "Model", ylabel = "Average Energy Jump Distance",
		legend = :topright,
		yticks = [0, 10, 20, 30], 
		color = :auto,  # Ensure distinct colors per group
		margin = 3Plots.mm,
		xticks = (collect(values(model_mapping)), keys(model_mapping)),
	)
	vline!(p, [100, 300, 500, 700], linestyle = :dash, color = :grey, label = false)
	savefig("icml2025/plots/energy_jump.png")
	
	# the acceptance rate plot
	df_filtered_no_ac = filter(row -> !(isnan(row.acceptance_prob)), df) # need to remove 0 cost (alg fails)
	df_summary = combine(groupby(df_filtered_no_ac, [:model, :explorer])) do subdf
		q25, med, q75 = quantile(subdf.acceptance_prob, [0.25, 0.5, 0.75])
		(med = med, q25 = q25, q75 = q75)
	end
	df_summary.error_low = df_summary.med .- df_summary.q25
	df_summary.error_high = df_summary.q75 .- df_summary.med
	df_summary.model_index = getindex.(Ref(model_mapping), df_summary.model)
	df_summary.explorer_shift = [explorer_mapping[e] for e in df_summary.explorer]
	df_summary.position = df_summary.model_index .+ df_summary.explorer_shift
	p = @df df_summary StatsPlots.scatter(
		:position, :med, group = :explorer,
		yerror = (df_summary.error_low, df_summary.error_high),  # IQR range
		markersize = 8, markerstrokewidth = 1.5, marker = :circle,  # Large dots
		xlabel = "Model", ylabel = "Average Acceptance Rate",
		legend = :bottomleft,
		yticks = [0, 0.25, 0.5, 0.75, 1.0], 
		color = :auto,  # Ensure distinct colors per group
		margin = 3Plots.mm,
		xticks = (collect(values(model_mapping)), keys(model_mapping)),
	)
	vline!(p, [100, 300, 500, 700], linestyle = :dash, color = :grey, label = false)
	savefig("icml2025/plots/accept_rate.png")
end

df = CSV.read("icml2025/exp_results_new.csv", DataFrame)

comparison_plots(df)