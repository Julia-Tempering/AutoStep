const Explorers = Union{SimpleRWMH, HitAndRunSlicer}

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
