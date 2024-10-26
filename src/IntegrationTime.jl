abstract type IntegrationTime end

"""
$SIGNATURES

Implements randomized integration time for HMC.

$FIELDS
"""
@kwdef struct AdaptiveRandomIntegrationTime{TF <: Real, TJitter <: Distributions.UnivariateDistribution, TI <: Integer} <: IntegrationTime
    """
    Integration time targeted by the sampler.
    """
    target_int_time::TF = 1.0 # TODO: this is no longer used (we directly manage base_n_leaps). Remove?

    """
    A `Distributions.UnivariateDistribution` used to jitter the integration time at each step.
    """
    jitter_dist::TJitter = Distributions.Uniform()

    """
    Thresholds on the autocorrelation of the energy used for tuning.
    """
    energy_ac1_thresholds::NTuple{2,TF} = (0.95, 0.99)

    """
    Skip first `adapt_after_round` rounds to adapt integration time.
    """
    adapt_after_round::TI = 3

    """
    Internal storage for `target_int_time/step_size`.
    """
    base_n_leaps::TF = 1.0

    """
    Limit on base_n_leaps.
    """
    max_base_n_leaps::TF = 1024.0 # Reason: implied default limit on n_leaps for NUTS (treedepth<=10).
end

function get_n_leaps(arit::AdaptiveRandomIntegrationTime{A,B,C}, rng::AbstractRNG, dim::Integer) where {A,B,C}
    max(one(C), round(C, arit.base_n_leaps * rand(rng, arit.jitter_dist) ) )
end
function adapt_integration_time(
    arit::AdaptiveRandomIntegrationTime,
    reduced_recorders,
    current_pt,
    _
    )
    current_pt.shared.iterators.round ≤ arit.adapt_after_round && return arit
    
    # only start adapting when acceptance rate stabilizes
    # NB: autoMALA acc rate is close to 0.65 for iid highdim targets
    min_acc_prob = minimum(Pigeons.recorder_values(reduced_recorders.explorer_acceptance_pr))
    min_acc_prob > 0.5 || return arit
    
    # try computing maximum AC of logpotential across chains
    max_cor = try
        maximum(Pigeons.energy_ac1s(reduced_recorders, true, current_pt))
    catch e
        e isa DomainError ? NaN : rethrow(e)
    end
    
    # bail if estimator is broken
    (0 ≤ max_cor ≤ 1) || return arit

    # adapt base_n_leaps
    new_base_n_leaps = arit.base_n_leaps
    if max_cor > last(arit.energy_ac1_thresholds)
        new_base_n_leaps = min(new_base_n_leaps * 2, arit.max_base_n_leaps)
    elseif max_cor < first(arit.energy_ac1_thresholds)
        new_base_n_leaps /= 2
    end
    # @info "max_cor=$max_cor\tnew_base_n_leaps=$new_base_n_leaps"

    return AdaptiveRandomIntegrationTime(
        arit.target_int_time, arit.jitter_dist, arit.energy_ac1_thresholds,
        arit.adapt_after_round, new_base_n_leaps, arit.max_base_n_leaps
    )
end

add_int_time_recorder!(recorders, ::AdaptiveRandomIntegrationTime) =
    push!(recorders, Pigeons.energy_ac1) # for adaptation

"""
$SIGNATURES

Convenience constructor for a non-random version of [`AdaptiveRandomIntegrationTime`](@ref).
"""
AdaptiveFixedIntegrationTime(; kwargs...) = 
    AdaptiveRandomIntegrationTime(;jitter_dist=Distributions.Dirac(1), kwargs...)

"""
$SIGNATURES

Convenience constructor for a non-random, non-adaptive version of [`AdaptiveRandomIntegrationTime`](@ref).
"""
FixedIntegrationTime(; kwargs...) = 
    AdaptiveFixedIntegrationTime(;adapt_after_round=typemax(Int), kwargs...)
