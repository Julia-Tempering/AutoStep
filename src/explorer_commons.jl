const Explorers = Union{SimpleRWMH, HitAndRunSlicer, SimpleAHMC}

Pigeons.step!(explorer::Explorers, replica, shared) =
    Pigeons.step!(explorer, replica, shared, replica.state)

function Pigeons.step!(explorer::Explorers, replica, shared, state::AbstractVector)
    log_potential = Pigeons.find_log_potential(replica, shared.tempering, shared)
    _extract_commons_and_run!(explorer, replica, shared, log_potential, state)
end

#=
Functions duplicated from PigeonsBridgeStanExt
=#

Pigeons.step!(explorer::Explorers, replica, shared, state::Pigeons.StanState) =
    Pigeons.step!(explorer, replica, shared, state.unconstrained_parameters)

#=
Functions duplicated from PigeonsDynamicPPLExt
=#

function Pigeons.step!(explorer::Explorers, replica, shared, vi::DynamicPPL.TypedVarInfo)
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


#=
step size dynamics that is common to autoRWMH and autoHMC
=#
function grow_step_size(log_joint_difference, step_size, selector, selector_params)
    n = 1
    while true
        step_size *= 2.0
        diff = log_joint_difference(step_size)
        if !isfinite(diff) || !should_grow(selector, selector_params, diff)
            return n, n - 1 # one less step, to avoid a potential cliff-like drop in acceptance
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
            return n, -n
        end
        n += 1
    end
end

# all are GroupBy(Int, Mean()) but saves me directly importing OnlineStats
as_factors() = Pigeons.am_factors()
abs_exponent_diff() = Pigeons.explorer_acceptance_pr()
energy_jump_distance() = Pigeons.explorer_acceptance_pr()
jitter_proposal_log_diff() = Pigeons.explorer_acceptance_pr()
num_doubling() = Pigeons.explorer_acceptance_pr()
