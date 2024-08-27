abstract type StepJitterAdaptationStrategy end
struct AdaptativeStepJitter <: StepJitterAdaptationStrategy end
struct FixedStepJitter <: StepJitterAdaptationStrategy end

@kwdef struct StepJitter{TDist <: UnivariateDistribution, TAdapt <: StepJitterAdaptationStrategy}
    """
    Distribution for drawing a random jitter (in log₂ space) of the deterministic
    autoRWMH step size.
    """
    dist::TDist = Normal(0, 0.5)

    """
    An adaptation strategy.
    """
    adapt_strategy::TAdapt = AdaptativeStepJitter()
end

# jitter adaptation
adapt_step_jitter(sj::StepJitter, _) = sj
function adapt_step_jitter(sj::StepJitter{<:Normal,AdaptativeStepJitter}, abs_exponent_diff)
    d = sj.dist
    new_sd = 0.5 * mean(Pigeons.recorder_values(abs_exponent_diff))
    # @info "Old jitter SD = $(round(d.σ, digits=4)) /// New jitter SD = $(round(new_sd, digits=4))"
    StepJitter(Normal(mean(d), new_sd), sj.adapt_strategy)
end
