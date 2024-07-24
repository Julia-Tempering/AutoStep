@testset "Invariance tests" begin
    # cannot use the one in Pigeons because conditioning does not work when the observation is an argument
    DynamicPPL.@model function product_of_probs(n_trials)
        p1 ~ Uniform()
        p2 ~ Uniform()
        n_successes ~ Binomial(n_trials, p1*p2)
        return n_successes
    end

    model = product_of_probs(100)
    target = TuringLogPotential(model)
    rng = SplittableRandom(1)

    step_selectors = (autoRWMH.MHSelector(), autoRWMH.MHSelectorLegacy(), autoRWMH.MHSelectorInverted())
    step_jitter_dists = (Dirac(0.0), Normal())
    foreach(Iterators.product(step_selectors, step_jitter_dists)) do (sss, jdist)
        explorer = SimpleRWMH(step_size_selector = sss, step_jitter_dist = jdist)
        @show explorer
        @test first(Pigeons.invariance_test(target, explorer, rng; condition_on=(:n_successes,)))
    end
end
