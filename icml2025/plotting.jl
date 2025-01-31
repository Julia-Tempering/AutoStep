using StatsPlots, CSV, DataFrames
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
for model in ["prostate"] # 
    for explorer in ["adaptive_rwmh", "nuts", "slicer"] #"adaptive_mala", "automala", "drhmc", "autorwmh", 
        draw_pairplot(model, 1, explorer)
    end
end
draw_pairplot("prostate", 10, "autorwmh")

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
	df_filtered = filter(row -> !isnan(row.miness), df) # need to remove NaN
	df_filtered.miness_per_sec = df_filtered.miness ./ df_filtered.time
	df_filtered.miness_per_cost = df_filtered.miness ./ df_filtered.cost
	df.minKSess_per_sec = df.minKSess ./ df.time

	sort!(df, :model) # ensure ordering on x-axis
	# minESS and minKSess just for the reference
	@df df_filtered StatsPlots.groupedboxplot(:model, :miness, group = :explorer, xlabel = "Model", ylabel = "minESS",
		legend = :bottomleft, color = :auto, yaxis = :log10)
	savefig("icml2025/plots/miness.png")
	@df df StatsPlots.groupedboxplot(:model, :minKSess, group = :explorer, xlabel = "Model", ylabel = "min KSESS",
		legend = :bottomleft, color = :auto, yaxis = :log10)
	savefig("icml2025/plots/minKSess.png")
	# Create the grouped boxplot for minESS/sec, minESS/cost, minKSess/sec, minKSess/cost
	@df df_filtered StatsPlots.groupedboxplot(:model, :miness_per_sec, group = :explorer, xlabel = "Model", ylabel = "minESS / sec",
		legend = :bottomleft, color = :auto, yaxis = :log10)
	savefig("icml2025/plots/miness_per_sec.png")
	@df df_filtered StatsPlots.groupedboxplot(:model, :miness_per_cost, group = :explorer, xlabel = "Model", ylabel = "minESS / cost",
		legend = :bottomleft, color = :auto, yaxis = :log10)
	savefig("icml2025/plots/miness_per_cost.png")
	@df df StatsPlots.groupedboxplot(:model, :minKSess_per_sec, group = :explorer, xlabel = "Model", ylabel = "min KSESS / sec",
		legend = :bottomleft, color = :auto, yaxis = :log10)
	savefig("icml2025/plots/minKSess_per_sec.png")

    df_filtered_no_cost = filter(row -> !(row.cost == 0), df) # need to remove 0 cost (alg fails)
	df_filtered_no_cost.minKSess_per_cost = df_filtered_no_cost.minKSess ./ df_filtered_no_cost.cost
	@df df_filtered_no_cost StatsPlots.groupedboxplot(:model, :minKSess_per_cost, group = :explorer, xlabel = "Model", ylabel = "min KSESS / cost",
		legend = :bottomleft, color = :auto, yaxis = :log10)
	savefig("icml2025/plots/minKSess_per_cost.png")

	# the energy jump plot
    df_filtered_no_energy = filter(row -> !(row.energy_jump_dist == 0 || isnan(row.energy_jump_dist) || !isfinite(row.energy_jump_dist)), df) # need to remove 0 cost (alg fails)
	@df df_filtered_no_energy StatsPlots.groupedboxplot(:model, :energy_jump_dist, group = :explorer, xlabel = "Model",
		ylabel = "Average Energy Jump Distance", legend = :topleft, color = :auto) # , ylim = (-0.03, 0.8)
	savefig("icml2025/plots/energy_jump.png")
	# the acceptance rate plot
    df_filtered_no_ac = filter(row -> !(isnan(row.acceptance_prob)), df) # need to remove 0 cost (alg fails)
	@df df_filtered_no_ac StatsPlots.groupedboxplot(:model, :acceptance_prob, group = :explorer, xlabel = "Model",
		ylabel = "Average Acceptance Rate", legend = :topleft, color = :auto, ylim = (-0.03, 1.2))
	savefig("icml2025/plots/accept_rate.png")
end

df1 = CSV.read("icml2025/exp_results_funnel2.csv", DataFrame)
df2 = CSV.read("icml2025/exp_results_funnel100.csv", DataFrame)
df3 = CSV.read("icml2025/exp_results_kilpisjarvi.csv", DataFrame)
df4 = CSV.read("icml2025/exp_results_orbital.csv", DataFrame)
df = vcat(df, df2, df3, df4)

comparison_plots(df)