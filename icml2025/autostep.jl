include("utils.jl")

function pt_sample_from_model(model, seed, my_explorer, n_rounds)
    round = 1 # NB: cannot start from >1 otherwise we miss the explorer n_steps from all but last round
    recorders = [
        record_default(); Pigeons.explorer_acceptance_pr; Pigeons.traces;
        Pigeons.reversibility_rate; online
    ]
    explorer = if my_explorer == "AutoStep RWMH"
        SimpleRWMH()
    elseif my_explorer == "AutoStep MALA"
        SimpleAHMC(int_time = AutoStep.FixedIntegrationTime())
    elseif my_explorer == "HitAndRunSlicer"
        HitAndRunSlicer()
    end
    pt = PT(Inputs(
        target      = model_to_target(model), 
        seed        = seed,
        n_rounds    = round,
        n_chains    = 1, 
        record      = recorders,
        explorer    = explorer, 
        show_report = true
    ))

    # run until minESS threshold is breached
    n_logprob = n_steps = n_samples = 0
    miness = 0.0
    local samples
    while round ≤ n_rounds # bail after this point
        pt = pigeons(pt)
        n_steps += first(Pigeons.explorer_n_steps(pt))
        samples = get_sample(pt) # only from last round
        n_samples = length(samples)
        n_logprob += if explorer isa SimpleRWMH || explorer isa HitAndRunSlicer
            n_steps # n_steps record the log potential evaluation for non-gradient based samplers
            else
                first(Pigeons.recorder_values(pt, :explorer_n_logprob))
            end
        pt = Pigeons.increment_n_rounds!(pt, 1)
        round += 1 
    end
    miness = min_ess_all_methods(samples, model)
    minKSess = min_KSess(samples, model)
    mean_1st_dim = first(mean(pt))
    var_1st_dim = first(var(pt))
    step_size = if explorer isa SliceSampler
        pt.shared.explorer.w
    elseif explorer isa HitAndRunSlicer
        pt.shared.explorer.slicer.w
    else
        pt.shared.explorer.step_size
    end
    energy_jump_dist = first(Pigeons.recorder_values(pt, :energy_jump_distance))
    time = sum(pt.shared.reports.summary.last_round_max_time) # despite name, it is a vector of time elapsed for all rounds
    acceptance_prob = explorer isa SliceSampler ? zero(miness) : 
        first(Pigeons.recorder_values(pt, :explorer_acceptance_pr))
    jitter_std = isa(pt.shared.explorer, HitAndRunSlicer) ? 0 :
        isa(pt.shared.explorer.step_jitter.dist, Normal) ? std(pt.shared.explorer.step_jitter.dist) : 0
    stats_df = DataFrame(
        explorer = my_explorer, model = model, 
        mean_1st_dim = mean_1st_dim, var_1st_dim = var_1st_dim, time=time, jitter_std = jitter_std, n_logprob = n_logprob, 
        n_steps=n_steps, miness=miness, minKSess = minKSess, acceptance_prob=acceptance_prob, step_size=step_size, 
        n_rounds = n_rounds, energy_jump_dist = energy_jump_dist)
    return samples, stats_df
end

pt_sample_from_model("funnel2", 1, "AutoStep RWMH", 15)
pt_sample_from_model("funnel2", 1, "AutoStep MALA", 15)
pt_sample_from_model("funnel2", 1, "HitAndRunSlicer", 15)