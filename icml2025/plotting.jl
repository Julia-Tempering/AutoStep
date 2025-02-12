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
	df_filtered = filter(row -> !isnan(row.miness), df) # need to remove NaN
	df_filtered.miness_per_sec = df_filtered.miness ./ df_filtered.time
	df_filtered.miness_per_cost = df_filtered.miness ./ df_filtered.cost
	df.minKSess_per_sec = df.minKSess ./ df.time

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


	# minESS
	df_summary = combine(groupby(df_filtered, [:model, :explorer])) do subdf
		q25, med, q75 = quantile(subdf.miness, [0.25, 0.5, 0.75])
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
		xlabel = "Model", ylabel = "minESS",
		yaxis = :log10, legend = :bottomleft,
		yticks = 10 .^ range(floor(log10(minimum(df_summary.q25))), ceil(log10(maximum(df_summary.q75))), length = 6),  # Auto log ticks
		color = :auto,  # Ensure distinct colors per group
		margin = 3Plots.mm,
		xticks = (collect(values(model_mapping)), keys(model_mapping)),
	)
	vline!(p, [100, 300, 500, 700], linestyle = :dash, color = :grey, label = false)
	savefig("icml2025/plots/miness.png")

	# min KSESS
	df_summary = combine(groupby(df, [:model, :explorer])) do subdf
		q25, med, q75 = quantile(subdf.minKSess, [0.25, 0.5, 0.75])
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
		xlabel = "Model", ylabel = "min KSESS",
		yaxis = :log10, legend = :topright,
		yticks = 10 .^ range(floor(log10(minimum(df_summary.q25))), ceil(log10(maximum(df_summary.q75))), length = 6),  # Auto log ticks
		color = :auto,  # Ensure distinct colors per group
		margin = 3Plots.mm,
		xticks = (collect(values(model_mapping)), keys(model_mapping)),
	)
	vline!(p, [100, 300, 500, 700], linestyle = :dash, color = :grey, label = false)
	savefig("icml2025/plots/minKSess.png")

	# Create the grouped boxplot for minESS/sec, minESS/cost, minKSess/sec, minKSess/cost
	# minESS / sec
	df_summary = combine(groupby(df_filtered, [:model, :explorer])) do subdf
		q25, med, q75 = quantile(subdf.miness_per_sec, [0.25, 0.5, 0.75])
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
		xlabel = "Model", ylabel = "minESS / sec",
		yaxis = :log10, legend = :bottomleft,
		yticks = 10 .^ range(floor(log10(minimum(df_summary.q25))), ceil(log10(maximum(df_summary.q75))), length = 6),  # Auto log ticks
		color = :auto,  # Ensure distinct colors per group
		margin = 3Plots.mm,
		xticks = (collect(values(model_mapping)), keys(model_mapping)),
	)
	vline!(p, [100, 300, 500, 700], linestyle = :dash, color = :grey, label = false)
	savefig("icml2025/plots/miness_per_sec.png")
	
	# minESS / cost
	df_summary = combine(groupby(df_filtered, [:model, :explorer])) do subdf
		q25, med, q75 = quantile(subdf.miness_per_cost, [0.25, 0.5, 0.75])
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
		xlabel = "Model", ylabel = "minESS / cost",
		yaxis = :log10, legend = :bottomleft,
		yticks = 10 .^ range(floor(log10(minimum(df_summary.q25))), ceil(log10(maximum(df_summary.q75))), length = 6),  # Auto log ticks
		color = :auto,  # Ensure distinct colors per group
		margin = 3Plots.mm,
		xticks = (collect(values(model_mapping)), keys(model_mapping)),
	)
	vline!(p, [100, 300, 500, 700], linestyle = :dash, color = :grey, label = false)
	savefig("icml2025/plots/miness_per_cost.png")

	# min KSESS / sec
	# Compute summary statistics: median, 25th percentile, and 75th percentile
	df_summary = combine(groupby(df, [:model, :explorer])) do subdf
		q25, med, q75 = quantile(subdf.minKSess_per_sec, [0.25, 0.5, 0.75])
		(med = med, q25 = q25, q75 = q75)
	end
	# Calculate error bars (distance from median to Q25 and Q75)
	df_summary.error_low = df_summary.med .- df_summary.q25
	df_summary.error_high = df_summary.q75 .- df_summary.med
	# Map model strings to numerical indices
	df_summary.model_index = getindex.(Ref(model_mapping), df_summary.model)
	# Define position shift for each explorer to avoid overlap
	df_summary.explorer_shift = [explorer_mapping[e] for e in df_summary.explorer]
	df_summary.position = df_summary.model_index .+ df_summary.explorer_shift
	# Plot median as large dots with vertical IQR lines
	p = @df df_summary StatsPlots.scatter(
		:position, :med, group = :explorer,
		yerror = (df_summary.error_low, df_summary.error_high),  # IQR range
		markersize = 8, markerstrokewidth = 1.5, marker = :circle,  # Large dots
		xlabel = "Model", ylabel = "min KSESS / sec",
		yaxis = :log10, legend = :topright,
		yticks = 10 .^ range(floor(log10(minimum(df_summary.q25))), ceil(log10(maximum(df_summary.q75))), length = 6),  # Auto log ticks
		color = :auto,  # Ensure distinct colors per group
		margin = 3Plots.mm,
		xticks = (collect(values(model_mapping)), keys(model_mapping)),
	)
	# Add vertical dashed lines
	vline!(p, [100, 300, 500, 700], linestyle = :dash, color = :grey, label = false)
	# Save the figure
	savefig("icml2025/plots/minKSess_per_sec.png")

	# min KSESS / cost
	df_filtered_no_cost = filter(row -> !(row.cost == 0), df) # need to remove 0 cost (alg fails)
	df_filtered_no_cost.minKSess_per_cost = df_filtered_no_cost.minKSess ./ df_filtered_no_cost.cost
	df_summary = combine(groupby(df_filtered_no_cost, [:model, :explorer])) do subdf
		q25, med, q75 = quantile(subdf.minKSess_per_cost, [0.25, 0.5, 0.75])
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
		xlabel = "Model", ylabel = "min KSESS / cost",
		yaxis = :log10, legend = :topright,
		yticks = 10 .^ range(floor(log10(minimum(df_summary.q25))), ceil(log10(maximum(df_summary.q75))), length = 6),  # Auto log ticks
		color = :auto,  # Ensure distinct colors per group
		margin = 3Plots.mm,
		xticks = (collect(values(model_mapping)), keys(model_mapping)),
	)
	vline!(p, [100, 300, 500, 700], linestyle = :dash, color = :grey, label = false)
	savefig("icml2025/plots/minKSess_per_cost.png")

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
	# @df df_filtered_no_ac StatsPlots.groupedboxplot(:model, :acceptance_prob, group = :explorer, xlabel = "Model",
	# 	ylabel = "Average Acceptance Rate", legend = :outerright, color = :auto, ylim = (-0.03, 1.02),
	# 	margin = 3Plots.mm)
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

df1 = CSV.read("icml2025/exp_results_funnel2.csv", DataFrame)
df2 = CSV.read("icml2025/exp_results_funnel100.csv", DataFrame)
df3 = CSV.read("icml2025/exp_results_kilpisjarvi.csv", DataFrame)
df4 = CSV.read("icml2025/exp_results_orbital.csv", DataFrame)
df5 = CSV.read("icml2025/exp_results_mRNA.csv", DataFrame)
# df6 = CSV.read("icml2025/exp_results_prostate.csv", DataFrame)
df = vcat(df1, df2, df3, df4, df5)

comparison_plots(df)