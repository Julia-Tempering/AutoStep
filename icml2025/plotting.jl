using StatsPlots, CSV, DataFrames

# ratio of running time of gradient VS log potential
# computed separately, recording the avg of time_gradient/time_log_prob
function log_prob_gradient_ratio(model::AbstractString)
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
    df = prepare_df(get_summary_df(experiment))
    plots_path = joinpath(base_dir(), "deliverables", experiment)

    sort!(df, :model) # ensure ordering on x-axis
    # Create the grouped boxplot for minESS/sec
    @df df StatsPlots.groupedboxplot(:model, :miness_per_sec, group=:sampler_type, xlabel="Model", ylabel="minESS / sec", 
        legend=:bottomleft, color=:auto, yaxis=:log10)
    savefig(joinpath(plots_path,"miness_per_sec_comparison.png"))

    # now create the minESS/cost plot
    @df df StatsPlots.groupedboxplot(:model, :miness_per_cost, group=:sampler_type, xlabel="Model", ylabel="minESS / cost", 
        legend=:bottomleft, color=:auto, yaxis=:log10)
    savefig(joinpath(plots_path,"miness_per_cost_comparison.png"))

    # now create the energy jump plot
    @df df StatsPlots.groupedboxplot(:model, :energy_jump_dist, group=:sampler_type, xlabel="Model", 
    ylabel="Average Energy Jump Distance", legend=:topleft, color=:auto, ylim= (-0.03,0.8))
    savefig(joinpath(plots_path,"energy_jump_comparison.png"))
end