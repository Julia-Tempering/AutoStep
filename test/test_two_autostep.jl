# using Revise
# include("activate_test_env.jl")
# include("../icml2025/autostep2.jl")

using Test

function test_autostep_selector()
    # Define different test cases
    x_values = [[1.0, 1.0], [0.5, -0.5], [-1.0, 2.0]]
    z_values = [[0.1, 0.2], [0.5, 0.5], [0.3, -0.3]]
    a_values = [0.3, 0.5, 0.2]
    b_values = [0.6, 0.8, 1.0]
    
    theta0 = 1.0
    f = fRWMH
    target = logdens_model("funnel2", stan_data("funnel2"))
    sqrtdiagM = ones(2)

    for x in x_values, z in z_values, a in a_values, b in b_values
        
        # Compute j_autostep2
        j_autostep2, cost, grad_eval = μ(x, z, a, b, theta0, f, target, sqrtdiagM)

        # Compute j_autostep
        target_log_potential(x) = LogDensityProblems.logdensity(target, x)
        recorders = [record_default()]
        chain = 1 # temporarily removed; remember to recover simpleRWMH after this!!
        selector = AutoStep.ASSelectorInverted()
        selector_params = [log(a), log(b)]
        j_autostep = AutoStep.auto_rwmh_step_size(target_log_potential, x, z, recorders, chain, theta0, selector, selector_params)

        println("j_autostep2: $j_autostep2, j_autostep: $j_autostep")

        @test isapprox(j_autostep2, j_autostep, atol=1e-6)  # they should be approximately equal

    end
end

# test_autostep_selector()


function test_autostep()
    # Define different test cases
    x_values = [[1.0, 1.0], [0.5, -0.5], [-1.0, 2.0]]
    z_values = [[0.1, 0.2], [0.5, 0.5], [0.3, -0.3]]
    a_values = [0.3, 0.5, 0.2]
    b_values = [0.6, 0.8, 1.0]
    
    theta0 = 1.0
    f = fRWMH
    target = logdens_model("funnel2", stan_data("funnel2"))
    sqrtdiagMhat = ones(2)
    rng1 = MersenneTwister(1)
    rng2 = MersenneTwister(1)

    for x in x_values, z in z_values, a in a_values, b in b_values
        
        # Compute autostep2
        new_state2, logacc, ejump, cost, theta, grad_eval = auto_step(x, f, theta0, target, sqrtdiagMhat, rng1)

        # Compute j_autostep
        target_log_potential(x) = LogDensityProblems.logdensity(target, x)
        explorer = SimpleRWMH(step_jitter = AutoStep.StepJitter(Dirac(0), AutoStep.FixedStepJitter()),
                        preconditioner = Pigeons.IdentityPreconditioner())
        recorders = [record_default()]
        chain = 1 # temporarily removed; remember to recover simpleRWMH after this!!
        new_state1 = AutoStep.auto_rwmh!(rng2, explorer, target_log_potential, x, recorders, chain, true)

        println("new_state_pigeon: $new_state1, new_state: $new_state2")

        @test isapprox(new_state1, new_state2, atol=1e-6)  # they should be approximately equal

    end
end

test_autostep()

