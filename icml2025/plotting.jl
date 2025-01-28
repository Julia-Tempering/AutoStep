using StatsPlots, CSV, DataFrames

# ratio of running time of gradient VS log potential
# computed separately, recording the avg of time_gradient/time_log_prob
function log_prob_gradient_ratio(model::AbstractString)
    # TODO: missing models
    if startswith(model, "horseshoe")
        35.66540160529861 # another run: 35.45077959104718
    elseif startswith(model, "mRNA")
        5.766575585884267 # 5.793217015357431
    elseif startswith(model, "kilpisjarvi")
        5.956119530847897 # 6.0015265040396
    elseif startswith(model, "funnel100") 
        65.43722654676135 # 63.34395522158833!!!!!!!!!!!!!
    elseif startswith(model, "funnel2")
        5.960239505901212 # 5.916476226247743
    else
        throw(KeyError(model))
    end
end

#=
comparison of all autoMCMC samplers and NUTS; experiment = "post_db"
=#
df = 
function comparison_plots(df::DataFrame)
    # prepare dataframe
    df.cost = ifelse.(
		map(explorer -> explorer in ["AutoStep RWMH", "Adaptive RWMH", "HitAndRunSlicer"], df.explorer),
		df.n_logprob,  # non gradient-based samplers
		df.n_logprob .+ 2 * df.n_steps .* log_prob_gradient_ratio.(df.model), # 1 leapfrog = 2 gradient eval
	) # gradient based: we use cost = #log_potential_eval + eta * #gradient_eval, where eta is model dependent
    df.miness_per_sec = df.miness ./ df.time
    df.miness_per_cost = df.miness ./ df.cost
    df.minKSess_per_sec = df.minKSess ./ df.time
    df.minKSess_per_cost = df.minKSess ./ df.cost

    sort!(df, :model) # ensure ordering on x-axis
    # Create the grouped boxplot for minESS/sec, minESS/cost, minKSess/sec, minKSess/cost
    @df df StatsPlots.groupedboxplot(:model, :miness_per_sec, group=:explorer, xlabel="Model", ylabel="minESS / sec", 
        legend=:bottomleft, color=:auto, yaxis=:log10)
    savefig("plots/miness_per_sec.png")
    @df df StatsPlots.groupedboxplot(:model, :miness_per_cost, group=:explorer, xlabel="Model", ylabel="minESS / cost", 
        legend=:bottomleft, color=:auto, yaxis=:log10)
    savefig("plots/miness_per_cost.png")
    @df df StatsPlots.groupedboxplot(:model, :minKSess_per_sec, group=:explorer, xlabel="Model", ylabel="min KSESS / sec", 
        legend=:bottomleft, color=:auto, yaxis=:log10)
    savefig("plots/minKSess_per_sec.png")
    @df df StatsPlots.groupedboxplot(:model, :minKSess_per_cost, group=:explorer, xlabel="Model", ylabel="min KSESS / cost", 
        legend=:bottomleft, color=:auto, yaxis=:log10)
    savefig("plots/minKSess_per_cost.png")

    # the energy jump plot
    @df df StatsPlots.groupedboxplot(:model, :energy_jump_dist, group=:explorer, xlabel="Model", 
    ylabel="Average Energy Jump Distance", legend=:topleft, color=:auto, ylim= (-0.03,0.8))
    savefig("plots/energy_jump.png")
    # the acceptance rate plot
    @df df StatsPlots.groupedboxplot(:model, :acceptance_prob, group=:explorer, xlabel="Model", 
    ylabel="Average Acceptance Rate", legend=:topleft, color=:auto, ylim= (-0.03,1.2))
    savefig("plots/accept_rate.png")
end