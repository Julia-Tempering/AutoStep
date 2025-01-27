"""
$SIGNATURES

A simple generalization of autoMALA towards autoHMC, where the number of
leapfrog steps is sampled independently of everything else. The autoMALA 
step-size selection procedure is then applied to HMC with the above number 
of leapfrog steps fixed.

$FIELDS
"""
@kwdef struct SimpleAHMC{T,TPrec <: Preconditioner,TIntTime <: IntegrationTime,TSSS <: StepSizeSelector,TJitter <: StepJitter}
    """
    An [`IntegrationTime`](@ref) specifying how to determine the number of leapfrog
    steps to carry out at each iteration.
    """
    int_time::TIntTime = AdaptiveRandomIntegrationTime()
  
    """
    A [`StepSizeSelector`](@ref) controlling the strategy for selecting the step size at each iteration.
    """
    step_size_selector::TSSS = ASSelectorInverted()
  
    """
    The number of steps (equivalently, momentum refreshments) between swaps.
    """
    n_refresh::Int = 1

    """
    The default backend to use for autodiff.
    See https://github.com/tpapp/LogDensityProblemsAD.jl#backends

    Certain targets may ignore it, e.g. if a manual differential is
    offered or when calling an external program such as Stan.
    """
    default_autodiff_backend::Symbol = :ForwardDiff

    """
    Starting point for the automatic step size algorithm.
    Gets updated automatically between each round.
    """
    step_size::Float64 = 1.0

    """
    A [`StepJitter`](@ref) object controlling the step size jitter.
    """
    step_jitter::TJitter = StepJitter()

    """
    A strategy for building a preconditioner.
    """
    preconditioner::TPrec = MixDiagonalPreconditioner()

    """
    This gets updated after first iteration; initially `nothing` in
    which case an identity mass matrix is used.
    """
    estimated_target_std_deviations::T = nothing

    # TODO: add option(s) for transformations? For now, doing it only for Turing
end

function Pigeons.adapt_explorer(explorer::SimpleAHMC, reduced_recorders, current_pt, new_tempering)
    # re-estimate std devs under the target; no adaptation for HMC/ MALA
    estimated_target_std_deviations = adapt_preconditioner(explorer.preconditioner, reduced_recorders)

    # new base stepsize = old_base_stepsize * 2 ^ median_of_j (use median for robustness against outliers)
    updated_step_size = explorer.step_size * 2.0 ^ median(Pigeons.recorder_values(reduced_recorders.step_size_exponent))
    println(updated_step_size)
    println(median(Pigeons.recorder_values(reduced_recorders.step_size_exponent)))

    # update integration time
    new_int_time = adapt_integration_time(
        explorer.int_time, reduced_recorders, current_pt, updated_step_size)

    # maybe adapt the jitter distribution based on observed average abs_exponent_diff
    updated_step_jitter = adapt_step_jitter(explorer.step_jitter, reduced_recorders.abs_exponent_diff)

    return SimpleAHMC(
        new_int_time, explorer.step_size_selector,
        explorer.n_refresh, explorer.default_autodiff_backend, updated_step_size,
        updated_step_jitter, explorer.preconditioner, estimated_target_std_deviations
    )
end

#=
Extract info common to all types of target and perform a step!()
=#
function _extract_commons_and_run!(explorer::SimpleAHMC, replica, shared, log_potential, state::AbstractVector)

    log_potential_autodiff = ADgradient(explorer.default_autodiff_backend, log_potential, replica)

    auto_hmc!(
        replica.rng,
        explorer,
        log_potential_autodiff,
        state,
        replica.recorders,
        replica.chain,
        # always use MH scheme
        true
    )
end

function auto_hmc!(
        rng::AbstractRNG,
        explorer::SimpleAHMC,
        target_log_potential,
        state::Vector,
        recorders,
        chain,
        use_mh_accept_reject)

    dim = length(state)

    momentum = get_buffer(recorders.buffers, :ah_momentum_buffer, dim)
    diag_precond = get_buffer(recorders.buffers, :ah_diag_precond, dim)
    build_preconditioner!(diag_precond, explorer.preconditioner, rng, explorer.estimated_target_std_deviations)
    start_state = get_buffer(recorders.buffers, :ah_state_buffer, dim)
    
    for i in 1:explorer.n_refresh
        # get a (possibly randomized) number of leapfrog steps from the IntegrationTime object
        n_leaps = get_n_leaps(explorer.int_time, rng, dim)

        # build augmented state
        start_state .= state
        randn!(rng, momentum)
        init_joint_log = log_joint(target_log_potential, state, momentum)
        @assert isfinite(init_joint_log) "SimpleAHMC can only be called on a configuration of positive density."

        # Draw bounds for the log acceptance ratio
        selector_params = draw_parameters(explorer.step_size_selector,rng)

        # compute the proposed step size
        proposed_exponent =
            auto_hmc_step_size(
                target_log_potential,
                diag_precond,
                state, momentum,
                recorders, chain,
                explorer.step_size, 
                explorer.step_size_selector,
                selector_params, n_leaps)
        proposed_jitter = rand(rng, explorer.step_jitter.dist)
        proposed_step_size = explorer.step_size * 2.0^(proposed_exponent+proposed_jitter)
        @record_if_requested!(recorders, :num_doubling, (chain, abs(proposed_exponent)))
        @record_if_requested!(recorders, :step_size_exponent, (chain, proposed_exponent))

        # move to proposed point
        hamiltonian_dynamics!(
            target_log_potential,
            diag_precond,
            state, momentum, proposed_step_size,
            n_leaps
        )
        final_joint_log = log_joint(target_log_potential, state, momentum)
        @record_if_requested!(recorders, :explorer_n_logprob, (chain, 4)) # two logprob eval: init_joint_log and final_joint_log
                                                                        # two more: one hamiltonian dynamics requires 2 logprob eval

        if !isfinite(final_joint_log) # check validity of new point (only relevant for nontrivial jitter)
            state .= start_state      # reject: go back to start state
            @record_if_requested!(recorders, :reversibility_rate, (chain, false))
            @record_if_requested!(recorders, :explorer_acceptance_pr, (chain, zero(final_joint_log)))
            @record_if_requested!(recorders, :energy_jump_distance, (chain, 0))
        elseif use_mh_accept_reject
            # flip
            momentum .*= -one(eltype(momentum))
            reversed_exponent =
                auto_hmc_step_size(
                    target_log_potential,
                    diag_precond,
                    state, momentum,
                    recorders, chain,
                    explorer.step_size, 
                    explorer.step_size_selector,
                    selector_params, n_leaps)
            reversibility_passed = reversed_exponent == proposed_exponent
            @record_if_requested!(recorders, :reversibility_rate, (chain, reversibility_passed))
            @record_if_requested!(recorders, :abs_exponent_diff, (chain, abs(proposed_exponent - reversed_exponent)))

            # compute the jitter z' needed to return to initial position
            # due to the involutive nature of the flipped leapfrog, this occurs iff
            #     eps0*2^(reversed_exponent+z') = eps0*2^(proposed_exponent+z)
            # <=> z' = (proposed_exponent+z)-reversed_exponent
            reversed_jitter = (proposed_exponent+proposed_jitter)-reversed_exponent
            jitter_proposal_log_diff = logpdf(explorer.step_jitter.dist, proposed_jitter) - 
                logpdf(explorer.step_jitter.dist, reversed_jitter)

            # compute acceptance probability and MH decision
            probability = if isfinite(jitter_proposal_log_diff)
                min(
                    one(final_joint_log), 
                    exp(final_joint_log - init_joint_log - jitter_proposal_log_diff)
                )
            else
                zero(final_joint_log)
            end

            @record_if_requested!(recorders, :explorer_acceptance_pr, (chain, probability))
            
            @record_if_requested!(recorders, :jitter_proposal_log_diff, (chain, jitter_proposal_log_diff))
            if rand(rng) < probability
                # accept: nothing to do, we work in-place
                @record_if_requested!(recorders, :energy_jump_distance, (chain, abs(final_joint_log - init_joint_log)))
            else
                # reject: go back to start state
                state .= start_state
                # no need to reset momentum as it will get resampled at beginning of the loop
                @record_if_requested!(recorders, :energy_jump_distance, (chain, 0))
            end
        end
    end
end

function auto_hmc_step_size(
        target_log_potential,
        diag_precond,
        state, momentum,
        recorders, chain,
        step_size, 
        selector, 
        selector_params, # should be the exact same in fwd and bwd pass!
        n_leaps)

    @assert step_size > 0

    log_joint_difference =
        hmc_log_joint_difference_function(
            target_log_potential,
            diag_precond,
            state, momentum,
            recorders, n_leaps)
    initial_difference = log_joint_difference(step_size)

    n_steps, exponent =
        if should_shrink(selector, selector_params, initial_difference)
            shrink_step_size(log_joint_difference, step_size, selector, selector_params)
        elseif should_grow(selector, selector_params, initial_difference)
            grow_step_size(log_joint_difference, step_size, selector, selector_params)
        else
            0, 0
        end

    @record_if_requested!(recorders, :explorer_n_steps, (chain, 1+n_steps*n_leaps)) # we do n_leaps leapfrogs per each grow/shrink step
    @record_if_requested!(recorders, :explorer_n_logprob, (chain, (1+n_steps)*(3+n_leaps))) # log_joint_diff computes logprob 3+n_leaps times
    @record_if_requested!(recorders, :as_factors, (chain, 2.0^exponent))
    return exponent
end


function hmc_log_joint_difference_function(
            target_log_potential,
            diag_precond,
            state, momentum,
            recorders,
            n_leaps)

    dim = length(state)

    state_before = get_buffer(recorders.buffers, :as_ljdf_state_before_buffer, dim)
    state_before .= state

    momentum_before = get_buffer(recorders.buffers, :as_ljdf_momentum_before_buffer, dim)
    momentum_before .= momentum

    h_before = log_joint(target_log_potential, state, momentum)
    function result(step_size)
        hamiltonian_dynamics!(
            target_log_potential, diag_precond,
            state, momentum, step_size, n_leaps)
        h_after = log_joint(target_log_potential, state, momentum)
        state .= state_before
        momentum .= momentum_before
        return h_after - h_before
    end
    return result
end



explorer_n_logprob() = Pigeons.explorer_n_steps() # reuse additive recorder

function Pigeons.explorer_recorder_builders(explorer::SimpleAHMC)
    result = [
        Pigeons.explorer_acceptance_pr,
        Pigeons.explorer_n_steps,
        as_factors,
        abs_exponent_diff,
        explorer_n_logprob,
        energy_jump_distance,
        jitter_proposal_log_diff,
        num_doubling,
        step_size_exponent
    ]
    gradient_based_sampler_recorders!(result, explorer)
    add_int_time_recorder!(result, explorer.int_time)
    return result
end

#=
Functions duplicated from GradientBasedSampler.jl
=#

function gradient_based_sampler_recorders!(recorders, explorer::SimpleAHMC)
    push!(recorders, Pigeons.buffers)
    push!(recorders, Pigeons.ad_buffers)
    if hasproperty(explorer, :preconditioner) && explorer.preconditioner isa Pigeons.AdaptedDiagonalPreconditioner
        push!(recorders, Pigeons._transformed_online) # for mass matrix adaptation
    end
end
