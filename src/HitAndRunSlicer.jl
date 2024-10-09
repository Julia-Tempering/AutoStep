"""
$SIGNATURES

Hit-and-run slice sampler.

$FIELDS
"""
@kwdef struct HitAndRunSlicer{T, TSS<:Pigeons.SliceSampler, TPrec<:Pigeons.Preconditioner}
    """
    A [`SliceSampler`](@ref) used to sample along rays.
    """
    slicer::TSS = Pigeons.SliceSampler(n_passes=1)

    """
    The number of steps (equivalently, direction refreshments) between swaps.
    """
    n_refresh::Int = 3

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

function Pigeons.adapt_explorer(explorer::HitAndRunSlicer, reduced_recorders, current_pt, new_tempering)
    new_slicer = Pigeons.adapt_explorer(explorer.slicer, reduced_recorders, current_pt, new_tempering)
    estimated_target_std_deviations = adapt_preconditioner(explorer.preconditioner, reduced_recorders)  
    HitAndRunSlicer(new_slicer, explorer.n_refresh, explorer.preconditioner, estimated_target_std_deviations)
end

#=
Extract info common to all types of target and perform a step!()
=#
function _extract_commons_and_run!(explorer::HitAndRunSlicer, replica, shared, log_potential, state::AbstractVector)
    vec_log_potential = vectorize(log_potential, replica)
    hit_and_run!(replica.rng, explorer, vec_log_potential, replica, state)
end

function hit_and_run!(
    rng::AbstractRNG,
    explorer::HitAndRunSlicer,
    target_log_potential,
    replica::Pigeons.Replica,
    state::AbstractVector
    )
    # get initial LP and check initial state is valid
    cached_lp_init = cached_lp = target_log_potential(state)
    @assert isfinite(cached_lp) "HitAndRunSlicer can only be called on a configuration of positive density."
    @record_if_requested!(replica.recorders, :explorer_n_steps, (replica.chain, 1)) # the only log_potential call here, the others are all by slicer

    # fetch buffers
    buffers = replica.recorders.buffers
    dim = length(state)
    diag_precond = get_buffer(buffers, :hrs_diag_precond, dim)
    direction = get_buffer(buffers, :hrs_direction, dim)

    # loop refreshments
    TElems = eltype(state)
    for _ in 1:explorer.n_refresh
        # get a (possibly randomized) preconditioner
        # note: this in 1/(std-deviation) scale
        build_preconditioner!(diag_precond, explorer.preconditioner, rng, explorer.estimated_target_std_deviations)

        # draw a direction N(0, Diag-Precond) (full conditional Gibbs move => always accepted)
        randn!(rng, direction)
        direction ./= diag_precond # divide because precond is inv std-dev scale

        # run slice sampling along the ray
        ray_lp, pointer = ray_lp_function(target_log_potential, state, direction, buffers)
        cached_lp = Pigeons.slice_sample_coord!(
            explorer.slicer,
            replica,
            pointer,
            ray_lp,
            cached_lp,
            TElems
        )

        # move to the new point
        step = pointer[]
        iszero(step) || hit_and_run_dynamics!(state, state, direction, step)
    end
    @record_if_requested!(replica.recorders, :energy_jump_distance, (replica.chain, abs(cached_lp_init - cached_lp)))
end

function ray_lp_function(
    target_log_potential,
    state::AbstractVector,
    direction::AbstractVector,
    buffers::Pigeons.Augmentation
    )
    proposed_state = get_buffer(buffers, :hrs_proposed_state, length(state))
    pointer = get_buffer(buffers, :hrs_pointer, 1) # works like a Ref (pointer[] === pointer[1]) but avoids re-allocating
    pointer[] = zero(eltype(pointer)) # to get correct initial state for the slicer, we need to start at 0
    function ray_lp(_)::eltype(state) # the state we get passed is actually the initial state, so it is useless
        hit_and_run_dynamics!(proposed_state, state, direction, pointer[])
        return target_log_potential(proposed_state)
    end
    return ray_lp, pointer
end

function hit_and_run_dynamics!(proposed_state, state, direction, step)
    @. proposed_state = state + step * direction
end

function Pigeons.explorer_recorder_builders(explorer::HitAndRunSlicer)
    result = Pigeons.explorer_recorder_builders(explorer.slicer)
    push!(result, energy_jump_distance)
    push!(result, Pigeons.buffers)
    if explorer.preconditioner isa Pigeons.AdaptedDiagonalPreconditioner
        push!(result, Pigeons._transformed_online) # for mass matrix adaptation
    end
    return result
end
