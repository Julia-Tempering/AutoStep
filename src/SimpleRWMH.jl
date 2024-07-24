## TODO: geometric mix of preconditioner; unadjusted move until pass the reverse-check
"""
$SIGNATURES

A generalization of autoMALA towards autoRWMH， with soft reversibility check 
and "inverted" step size selector.

$FIELDS
"""
@kwdef struct SimpleRWMH{T,TPrec <: Preconditioner,TSSS <: StepSizeSelector,TJitter <: UnivariateDistribution}
    """
    A [`StepSizeSelector`](@ref) controlling the strategy for selecting the step size at each iteration.
    """
    step_size_selector::TSSS = MHSelectorInverted()
  
    """
    Starting point for the automatic step size algorithm.
    Gets updated automatically between each round.
    """
    step_size::Float64 = 1.0

    """
    Distribution for drawing a random jitter (in log₂ space) of the deterministic
    autoRWMH step size.
    """
    step_jitter_dist::TJitter = Normal(0, 0.5) #Dirac(0.0)

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
    # re-estimate std devs under the target
    estimated_target_std_deviations = adapt_preconditioner(explorer.preconditioner, reduced_recorders)

    # use the mean across chains of the mean shrink/grow factor to compute a new baseline stepsize
    updated_step_size = explorer.step_size * mean(mean.(values(value(reduced_recorders.ar_factors))))

    return SimpleRWMH(
        explorer.step_size_selector, updated_step_size,
        explorer.step_jitter_dist,
        explorer.preconditioner, estimated_target_std_deviations
    )
end

#=
Extract info common to all types of target and perform a step!()
=#
function _extract_commons_and_run!(explorer::SimpleRWMH, replica, shared, log_potential, state::AbstractVector)
    vec_log_potential = vectorize(log_potential, replica)
    is_first_scan_of_round = shared.iterators.scan == 1

    auto_rwmh!(
        replica.rng,
        explorer,
        vec_log_potential,
        state,
        replica.recorders,
        replica.chain,
        # In the transient phase, the rejection rate for the
        # reversibility check can be high, so skip accept-rejct
        # for the initial scan of each round.
        # We only do this on the first scan of each round.
        # Since the number of iterations per round increases,
        # the fraction of time we do this decreases to zero.
        !is_first_scan_of_round
    )
end

# we add tricks to make it non-allocating
function random_walk_dynamics!(state, step_size, diag_precond)
    state .+= step_size .*diag_precond
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

    diag_precond = get_buffer(recorders.buffers, :ar_ones_buffer, dim)
    build_preconditioner!(diag_precond, explorer.preconditioner, rng, explorer.estimated_target_std_deviations)
    start_state = get_buffer(recorders.buffers, :ar_state_buffer, dim)
    random_walk = get_buffer(recorders.buffers, :ar_walk_buffer, dim)
    
    # build augmented state
    start_state .= state
    randn!(rng, random_walk)
    init_joint_log = target_log_potential(state)
    @assert isfinite(init_joint_log) "SimpleRWMH can only be called on a configuration of positive density."

    # Draw bounds for the log acceptance ratio
    selector_params = draw_parameters(explorer.step_size_selector,rng)

    # compute the proposed step size
    proposed_exponent =
        auto_step_size(
            target_log_potential,
            diag_precond,
            state, random_walk,
            recorders, chain,
            explorer.step_size, 
            explorer.step_size_selector,
            selector_params)
    proposed_step_size = explorer.step_size * 2.0^proposed_exponent
    proposed_jitter = rand(rng, explorer.step_jitter_dist)

    # move to proposed point
    random_walk_dynamics!(state, proposed_step_size, diag_precond)
    
    if use_mh_accept_reject
        # flip
        random_walk .*= -one(eltype(random_walk))
        reversed_exponent =
            auto_step_size(
                target_log_potential,
                diag_precond,
                state, random_walk,
                recorders, chain,
                explorer.step_size, 
                explorer.step_size_selector,
                selector_params)
            # compute the jitter z' needed to return to initial position
            #     eps0*2^(reversed_exponent+z') = eps0*2^(proposed_exponent+z)
            # <=> z' = proposed_exponent-reversed_exponent+z
        jitter_needed_to_return = proposed_exponent-reversed_exponent+proposed_jitter
        reversed_jitter_log_prob = logpdf(explorer.step_jitter_dist, jitter_needed_to_return)
        reversibility_passed = isfinite(reversed_jitter_log_prob)
        @record_if_requested!(recorders, :reversibility_rate, (chain, reversibility_passed))

        # compute acceptance probability and MH decision
        final_joint_log = target_log_potential(state)
        probability = if reversibility_passed
            min(one(final_joint_log),
                exp(final_joint_log - init_joint_log + reversed_jitter_log_prob - 
                    logpdf(explorer.step_jitter_dist, proposed_jitter)))
        else
            zero(final_joint_log)
        end
        @record_if_requested!(recorders, :explorer_acceptance_pr, (chain, probability))
        if rand(rng) < probability
            # accept: nothing to do, we work in-place
        else
            # reject: go back to start state
            state .= start_state
        end
    end
end

function auto_step_size(
        target_log_potential,
        diag_precond,
        state, random_walk,
        recorders, chain,
        step_size, 
        selector, 
        selector_params) # should be the exact same in fwd and bwd pass!

    @assert step_size > 0

    log_joint_difference =
        log_joint_difference_function(
            target_log_potential,
            diag_precond,
            state, random_walk,
            recorders)
    initial_difference = log_joint_difference(step_size)

    exponent =
        if should_shrink(selector, selector_params, initial_difference)
            shrink_step_size(log_joint_difference, step_size, selector, selector_params)
        elseif should_grow(selector, selector_params, initial_difference)
            grow_step_size(log_joint_difference, step_size, selector, selector_params)
        else
            0
        end

    @record_if_requested!(recorders, :ar_factors, (chain, 2.0^exponent))
    return exponent
end

function grow_step_size(log_joint_difference, step_size, selector, selector_params)
    n = 1
    while true
        step_size *= 2.0
        diff = log_joint_difference(step_size)
        if !isfinite(diff) || !should_grow(selector, selector_params, diff)
            return n - 1 # one less step, to avoid a potential cliff-like drop in acceptance
        end
        n += 1
    end
end

function shrink_step_size(log_joint_difference, step_size, selector, selector_params)
    n = 1
    while true
        step_size /= 2.0
        diff = log_joint_difference(step_size)
        #=
        Note that shrink is a bit different than grow.
        We do not assume here that the diff is necessarily
        finite when we start this loop: indeed, when the step
        size is too big, we may have to shrink several times
        until we get to a scale giving a finite evaluation.
        =#
        if step_size == 0.0
            error("Could not find an acceptable positive step size (selector_params: $selector_params")
        end
        if !should_shrink(selector, selector_params, diff)
            return -n
        end
        n += 1
    end
end


function log_joint_difference_function(
            target_log_potential,
            diag_precond,
            state, random_walk,
            recorders)

    dim = length(state)

    state_before = get_buffer(recorders.buffers, :ar_ljdf_state_before_buffer, dim)
    state_before .= state

    random_walk_before = get_buffer(recorders.buffers, :ar_ljdf_random_walk_before_buffer, dim)
    random_walk_before .= random_walk

    h_before = target_log_potential(state)
    function result(step_size)
        random_walk_dynamics!(state, step_size, diag_precond)
        h_after = target_log_potential(state)
        state .= state_before
        random_walk .= random_walk_before
        return h_after - h_before
    end
    return result
end


ar_factors() = Pigeons.am_factors()

function Pigeons.explorer_recorder_builders(explorer::SimpleRWMH)
    result = [
        Pigeons.explorer_acceptance_pr,
        ar_factors,
        Pigeons.buffers
    ]
    if explorer.preconditioner isa Pigeons.AdaptedDiagonalPreconditioner
        push!(result, Pigeons._transformed_online) # for mass matrix adaptation
    end
    return result
end

###############################################################################
# duplicated functions
###############################################################################

#=
Functions duplicated from GradientBasedSampler.jl
=#

Pigeons.step!(explorer::SimpleRWMH, replica, shared) =
    Pigeons.step!(explorer, replica, shared, replica.state)

function Pigeons.step!(explorer::SimpleRWMH, replica, shared, state::AbstractVector)
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    _extract_commons_and_run!(explorer, replica, shared, log_potential, state)
end

#=
Functions duplicated from PigeonsBridgeStanExt
=#

Pigeons.step!(explorer::SimpleRWMH, replica, shared, state::Pigeons.StanState) =
    Pigeons.step!(explorer, replica, shared, state.unconstrained_parameters)

#=
Functions duplicated from PigeonsDynamicPPLExt
=#

function Pigeons.step!(explorer::SimpleRWMH, replica, shared, vi::DynamicPPL.TypedVarInfo)
    vector_state = Pigeons.get_buffer(replica.recorders.buffers, :flattened_vi, get_dimension(vi))
    flatten!(vi, vector_state) # in-place DynamicPPL.getall
    Pigeons.step!(explorer, replica, shared, vector_state)
    DynamicPPL.setall!(replica.state, vector_state)
end

get_dimension(vi::DynamicPPL.TypedVarInfo) = sum(meta -> sum(length, meta.ranges), vi.metadata)

function flatten!(vi::DynamicPPL.TypedVarInfo, dest::Array)
    i = firstindex(dest)
    for meta in vi.metadata
        vals = meta.vals
        for r in meta.ranges
            N = length(r)
            copyto!(dest, i, vals, first(r), N)
            i += N
        end
    end
    return dest
end
