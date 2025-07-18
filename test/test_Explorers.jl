@testset "RWMH Invariance tests" begin
    rng = SplittableRandom(1)
    step_selectors = (AutoStep.ASNonAdaptiveSelector(), AutoStep.ASSelectorInverted())
    step_jitters = (AutoStep.StepJitter(dist=Dirac(0)), AutoStep.StepJitter(dist=Normal(0.0, 0.5)))
    # cannot use the one in Pigeons because conditioning does not work when the observation is an argument
    DynamicPPL.@model function product_of_probs(n_trials)
        p1 ~ Uniform()
        p2 ~ Uniform()
        n_successes ~ Binomial(n_trials, p1*p2)
        return n_successes
    end
    @model function funnel()
        y ~ Normal(0, 3)
        x ~ Normal(0, exp(y/2))
        z ~ Bernoulli()
    end
    targets = (TuringLogPotential(product_of_probs(100)), Pigeons.toy_mvn_target(10), TuringLogPotential(funnel()))
    kwargs = ( (;condition_on=(:n_successes,)), (;), (;condition_on=(:z,)) )
    @testset "$(nameof(typeof(target)))" for (target, kwarg) in zip(targets, kwargs)
        @show nameof(typeof(target))
        foreach(Iterators.product(step_selectors, step_jitters)) do (sss, sj)
            explorer = SimpleRWMH(step_size_selector = sss, step_jitter = sj, n_refresh=50)
            @show explorer
            @test first(Pigeons.invariance_test(target, explorer, rng; kwarg...))
        end
    end

    target = TuringLogPotential(product_of_probs(100))
    @testset "HitAndRunSlicer" begin
        explorer = HitAndRunSlicer(n_refresh=50)
        @show explorer
        @test first(Pigeons.invariance_test(target, explorer, rng; condition_on=(:n_successes,)))
        @test first(Pigeons.invariance_test(toy_mvn_target(10), explorer, rng))
    end
end

@testset "RWMH StepJitter" begin
    t = Pigeons.stan_funnel()
    explorer = SimpleRWMH()
    init_σ = explorer.step_jitter.dist.σ
    pt = pigeons(target = t, explorer = explorer, n_chains = 1)
    @test !(pt.shared.explorer.step_jitter.dist.σ ≈ init_σ)

    explorer = SimpleRWMH(step_jitter = AutoStep.StepJitter(adapt_strategy=AutoStep.FixedStepJitter()))
    pt = pigeons(target = t, explorer = explorer, n_chains = 1)
    @test pt.shared.explorer.step_jitter.dist.σ == init_σ
end

@testset "HMC StepJitter" begin
    t = Pigeons.stan_funnel()
    explorer = SimpleAHMC()
    init_σ = explorer.step_jitter.dist.σ
    pt = pigeons(target = t, explorer = explorer, n_chains = 1)
    @test !(pt.shared.explorer.step_jitter.dist.σ ≈ init_σ)

    explorer = SimpleAHMC(step_jitter = AutoStep.StepJitter(adapt_strategy=AutoStep.FixedStepJitter()))
    pt = pigeons(target = t, explorer = explorer, n_chains = 1)
    @test pt.shared.explorer.step_jitter.dist.σ == init_σ
end

@testset "HMC Invariance tests" begin
    rng = SplittableRandom(1)
    int_times = (AutoStep.AdaptiveRandomIntegrationTime(), AutoStep.FixedIntegrationTime(), AutoStep.AdaptiveFixedIntegrationTime())
    step_selectors = (AutoStep.ASNonAdaptiveSelector(), AutoStep.ASSelectorInverted())
    step_jitters = (AutoStep.StepJitter(dist=Dirac(0)), AutoStep.StepJitter(dist=Normal(0.0, 0.5)))

    DynamicPPL.@model function product_of_probs(n_trials)
        p1 ~ Uniform()
        p2 ~ Uniform()
        n_successes ~ Binomial(n_trials, p1*p2)
        return n_successes
    end
    @model function funnel()
        y ~ Normal(0, 3)
        x ~ Normal(0, exp(y/2))
        z ~ Bernoulli()
    end
    targets = (TuringLogPotential(product_of_probs(100)), Pigeons.toy_mvn_target(10), TuringLogPotential(funnel()))
    kwargs = ( (;condition_on=(:n_successes,)), (;), (;condition_on=(:z,)) )
    @testset "$(nameof(typeof(target)))" for (target, kwarg) in zip(targets, kwargs)
        @show nameof(typeof(target))
        foreach(Iterators.product(int_times, step_selectors, step_jitters)) do (itime, sss, sj)
            explorer = SimpleAHMC(
                n_refresh=10, int_time = itime, step_size_selector = sss, step_jitter = sj)
            @show explorer
            @test first(Pigeons.invariance_test(target, explorer, rng; kwarg...))
        end
    end
end

