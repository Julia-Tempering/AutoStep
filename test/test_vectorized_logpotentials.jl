@testset "Vectorized logpotentials" begin
    pt = pigeons(target = Pigeons.toy_stan_unid_target(), explorer=SimpleRWMH())

    for i in (1,5,10)
        replica = pt.replicas[i];
        int_lp = Pigeons.find_log_potential(replica, pt.shared.tempering, pt.shared)
        vint = AutoStep.vectorize(int_lp, replica)
        @test int_lp(replica.state) === vint(replica.state.unconstrained_parameters)
    end

    pt = pigeons(target = Pigeons.toy_turing_unid_target(), explorer=SimpleRWMH())

    for i in (1,5,10)
        replica = pt.replicas[i];
        int_lp = Pigeons.find_log_potential(replica, pt.shared.tempering, pt.shared)
        vint = AutoStep.vectorize(int_lp, replica)
        state = replica.state
        vector_state = Pigeons.get_buffer(replica.recorders.buffers, :flattened_vi, AutoStep.get_dimension(state))
        AutoStep.flatten!(state, vector_state) # in-place DynamicPPL.getall
        @test int_lp(state) === vint(vector_state)
    end
end
