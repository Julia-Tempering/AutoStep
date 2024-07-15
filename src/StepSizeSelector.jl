abstract type StepSizeSelector end

draw_parameters(::StepSizeSelector, rng::AbstractRNG) = extrema((-randexp(rng),-randexp(rng)))
should_grow(::StepSizeSelector, bounds, log_diff) = log_diff > last(bounds) # R > b
should_shrink(::StepSizeSelector, bounds, log_diff) = 
    !isfinite(log_diff) || log_diff < first(bounds) # R < a

"""
$SIGNATURES

autoMALA-style step size selector.
"""
struct MHSelector <: StepSizeSelector end

"""
$SIGNATURES

autoMALA-style step size selector with the original endpoint sampling procedure.
"""
struct MHSelectorLegacy <: StepSizeSelector end

function draw_parameters(::MHSelectorLegacy, rng::AbstractRNG)
    a = rand(rng)
    b = rand(rng)
    log(min(a, b)), log(max(a, b))
end

"""
$SIGNATURES

autoMALA-style step size selector with symmetric acceptance region.
"""
struct MHSelectorInverted <: StepSizeSelector end

should_grow(::MHSelectorInverted, bounds, log_diff) = 
    abs(log_diff) + last(bounds) < zero(log_diff) # |logR| < -log b
should_shrink(::MHSelectorInverted, bounds, log_diff) = 
    !isfinite(log_diff) || abs(log_diff) + first(bounds) > zero(log_diff) # |logR| > -log a
