## TODO: geometric mix of preconditioner; unadjusted move until pass the reverse-check
"""
$SIGNATURES

A generalization of autoMALA towards autoRWMH， with soft reversibility check 
and "inverted" step size selector.

$FIELDS
"""
@kwdef struct SimpleRWMH{T,TPrec <: Preconditioner,TSSS <: StepSizeSelector,TJitter <: StepJitter}
    """
    The number of steps (equivalently, direction refreshments) between swaps.
    """
    n_refresh::Int = 1

    """
    A [`StepSizeSelector`](@ref) controlling the strategy for selecting the step size at each iteration.
    """
    step_size_selector::TSSS = ASSelectorInverted()
  
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
end

function Pigeons.adapt_explorer(explorer::SimpleRWMH, reduced_recorders, current_pt, new_tempering)
    # re-estimate std devs under the target; no adaption for RWMH, keep hand tuning param
    estimated_target_std_deviations = adapt_preconditioner(explorer.preconditioner, reduced_recorders)

    # new base stepsize = old_base_stepsize * 2 ^ median_of_j (use median for robustness against outliers)
    updated_step_size = explorer.step_size * 2.0 ^ median(Pigeons.recorder_values(reduced_recorders.step_size_exponent))

    # maybe adapt the jitter distribution based on observed average abs_exponent_diff
    updated_step_jitter = adapt_step_jitter(explorer.step_jitter, reduced_recorders.abs_exponent_diff)

    return SimpleRWMH(
        explorer.n_refresh, explorer.step_size_selector, updated_step_size,
        updated_step_jitter, explorer.preconditioner, estimated_target_std_deviations
    )
end

#=
Extract info common to all types of target and perform a step!()
=#
function _extract_commons_and_run!(explorer::SimpleRWMH, replica, shared, log_potential, state::AbstractVector)
    vec_log_potential = vectorize(log_potential, replica)

    auto_rwmh!(
        replica.rng,
        explorer,
        vec_log_potential,
        state,
        replica.recorders,
        replica.chain,
        # always use MH scheme
        true
    )
end

# we add tricks to make it non-allocating
function random_walk_dynamics!(state, step_size, random_walk)
    state .+= step_size .* random_walk
    return true
end

function auto_rwmh!(
        rng::AbstractRNG,
        explorer::SimpleRWMH,
        target_log_potential,
        state::Vector,
        recorders,
        chain,
        use_mh_accept_reject)

    dim = length(state)

    diag_precond = get_buffer(recorders.buffers, :ar_diag_precond, dim)
    start_state = get_buffer(recorders.buffers, :ar_state_buffer, dim)
    random_walk = get_buffer(recorders.buffers, :ar_walk_buffer, dim)
    build_preconditioner!(diag_precond, explorer.preconditioner, rng, explorer.estimated_target_std_deviations)
    
    for _ in 1:explorer.n_refresh
        # Draw bounds for the log acceptance ratio
        a = rand(rng)
        b = rand(rng)
        selector_params = [log(min(a, b)), log(max(a, b))] # draw_parameters(explorer.step_size_selector,rng)
        # build augmented state
        start_state .= state
        randn!(rng, random_walk)
        random_walk .= random_walk ./ diag_precond # divide diag_precond because precond is inv std
        init_joint_log = target_log_potential(state)
        @assert isfinite(init_joint_log) "SimpleRWMH can only be called on a configuration of positive density."

        # compute the proposed step size
        proposed_exponent =
            auto_rwmh_step_size(
                target_log_potential,
                state, random_walk,
                recorders, chain,
                explorer.step_size, 
                explorer.step_size_selector,
                selector_params)
        @record_if_requested!(recorders, :num_doubling, (chain, abs(proposed_exponent))) #record number of doublings/halvings
        @record_if_requested!(recorders, :step_size_exponent, (chain, proposed_exponent)) #record number of doublings/halvings
        proposed_jitter = rand(rng, explorer.step_jitter.dist)
        proposed_step_size = explorer.step_size * 2.0^(proposed_exponent+proposed_jitter)

        # move to proposed point
        random_walk_dynamics!(state, proposed_step_size, random_walk)
        final_joint_log = target_log_potential(state)
        @record_if_requested!(recorders, :explorer_n_steps, (chain, 2)) # two logprob evaluations: final and init

        if !isfinite(final_joint_log) # check validity of new point (only relevant for nontrivial jitter)
            state .= start_state      # reject: go back to start state
            @record_if_requested!(recorders, :reversibility_rate, (chain, false))
            @record_if_requested!(recorders, :explorer_acceptance_pr, (chain, zero(final_joint_log)))
        elseif use_mh_accept_reject
            # flip
            random_walk .*= -one(eltype(random_walk))
            reversed_exponent =
                auto_rwmh_step_size(
                    target_log_potential,
                    state, random_walk,
                    recorders, chain,
                    explorer.step_size, 
                    explorer.step_size_selector,
                    selector_params)
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
            else
                # reject: go back to start state
                state .= start_state
            end
        end
        return state
        final_joint_log = target_log_potential(state)
        @record_if_requested!(recorders, :energy_jump_distance, (chain, abs(final_joint_log - init_joint_log)))
    end
end

function auto_rwmh_step_size(
        target_log_potential,
        state, random_walk,
        recorders, chain,
        step_size, 
        selector, 
        selector_params) # should be the exact same in fwd and bwd pass!

    @assert step_size > 0

    h_before = target_log_potential(state)
    log_joint_difference =
        rwmh_log_joint_difference_function(
            target_log_potential,
            state, random_walk,
            recorders, h_before)
    initial_difference = log_joint_difference(step_size)

    n_steps, exponent =
        if should_shrink(selector, selector_params, initial_difference)
            shrink_step_size(log_joint_difference, step_size, selector, selector_params)
        elseif should_grow(selector, selector_params, initial_difference)
            grow_step_size(log_joint_difference, step_size, selector, selector_params)
        else
            0, 0
        end
    # @record_if_requested!(recorders, :explorer_n_steps, (chain, 1+n_steps)) # every log_joint_difference call logprob once
    # @record_if_requested!(recorders, :as_factors, (chain, 2.0^exponent))
    return exponent
end


function rwmh_log_joint_difference_function(
            target_log_potential,
            state, random_walk,
            recorders, h_before)

    dim = length(state)

    # state_before = get_buffer(recorders.buffers, :as_ljdf_state_before_buffer, dim)
    state_before = zero(state)
    state_before .= state

    # random_walk_before = get_buffer(recorders.buffers, :as_ljdf_random_walk_before_buffer, dim)
    random_walk_before = zero(random_walk)
    random_walk_before .= random_walk

    function result(step_size)
        random_walk_dynamics!(state, step_size, random_walk)
        h_after = target_log_potential(state)
        state .= state_before
        random_walk .= random_walk_before
        return h_after - h_before
    end
    return result
end



function Pigeons.explorer_recorder_builders(explorer::SimpleRWMH)
    result = [
        Pigeons.explorer_acceptance_pr,
        Pigeons.explorer_n_steps,
        as_factors,
        Pigeons.buffers,
        abs_exponent_diff,
        energy_jump_distance,
        jitter_proposal_log_diff,
        num_doubling,
        step_size_exponent
    ]
    if explorer.preconditioner isa Pigeons.AdaptedDiagonalPreconditioner
        push!(result, Pigeons._transformed_online) # for mass matrix adaptation
    end
    return result
end
