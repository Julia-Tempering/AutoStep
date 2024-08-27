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

    @testset "SimpleRWMH" begin
        step_selectors = (autoRWMH.MHSelector(), autoRWMH.MHSelectorLegacy(), autoRWMH.MHSelectorInverted())
        step_jitters = (autoRWMH.StepJitter(dist=Dirac(0)), autoRWMH.StepJitter(dist=Normal(0.0, 0.5)))
        foreach(Iterators.product(step_selectors, step_jitters)) do (sss, sj)
            explorer = SimpleRWMH(step_size_selector = sss, step_jitter = sj, n_refresh=50)
            @show explorer
            @test first(Pigeons.invariance_test(target, explorer, rng; condition_on=(:n_successes,)))
            @test first(Pigeons.invariance_test(toy_mvn_target(10), explorer, rng))
        end 
    end

    @testset "HitAndRunSlicer" begin
        explorer = HitAndRunSlicer(n_refresh=50)
        @show explorer
        @test first(Pigeons.invariance_test(target, explorer, rng; condition_on=(:n_successes,)))
        @test first(Pigeons.invariance_test(toy_mvn_target(10), explorer, rng))
    end
end

@testset "StepJitter" begin
    t = Pigeons.stan_funnel()
    explorer = SimpleRWMH()
    init_σ = explorer.step_jitter.dist.σ
    pt = pigeons(target = t, explorer = explorer, n_chains = 1)
    @test !(pt.shared.explorer.step_jitter.dist.σ ≈ init_σ)

    explorer = SimpleRWMH(step_jitter = autoRWMH.StepJitter(adapt_strategy=autoRWMH.FixedStepJitter()))
    pt = pigeons(target = t, explorer = explorer, n_chains = 1)
    @test pt.shared.explorer.step_jitter.dist.σ == init_σ
end
