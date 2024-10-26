abstract type StepSizeSelector end

draw_parameters(::StepSizeSelector, rng::AbstractRNG) = extrema((-randexp(rng),-randexp(rng)))
should_grow(::StepSizeSelector, bounds, log_diff) = log_diff > last(bounds) # R > b
should_shrink(::StepSizeSelector, bounds, log_diff) = 
    !isfinite(log_diff) || log_diff < first(bounds) # R < a

"""
$SIGNATURES

autoMALA-style step size selector.
"""
struct ASSelector <: StepSizeSelector end

"""
$SIGNATURES

autoMALA-style step size selector with the original endpoint sampling procedure.
"""
struct ASSelectorLegacy <: StepSizeSelector end

function draw_parameters(::ASSelectorLegacy, rng::AbstractRNG)
    a = rand(rng)
    b = rand(rng)
    log(min(a, b)), log(max(a, b))
end

"""
$SIGNATURES

autoMALA-style step size selector with symmetric acceptance region.
"""
struct ASSelectorInverted <: StepSizeSelector end

should_grow(::ASSelectorInverted, bounds, log_diff) = 
    abs(log_diff) + last(bounds) < zero(log_diff) # |logR| < -log b
should_shrink(::ASSelectorInverted, bounds, log_diff) = 
    !isfinite(log_diff) || abs(log_diff) + first(bounds) > zero(log_diff) # |logR| > -log a

"""
$SIGNATURES

step size selector that does not automatically select step size
"""
struct ASNonAdaptiveSelector <: StepSizeSelector end

should_grow(::ASNonAdaptiveSelector, bounds, log_diff) = false
should_shrink(::ASNonAdaptiveSelector, bounds, log_diff) = false
