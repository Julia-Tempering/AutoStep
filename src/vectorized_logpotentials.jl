# By default, leave log potential as is
vectorize(lp, ::Pigeons.Replica) = lp

# Vectorization of TuringLogPotentials
vectorize(lp::Pigeons.TuringLogPotential, replica::Pigeons.Replica) =
    DynamicPPL.LogDensityFunction(replica.state, lp.model, lp.context)

"""
$SIGNATURES

A wrapper over an [`InterpolatedLogPotential`](@ref) that forces both endpoints
of its enclosed path to implement `LogDensityProblems.logdensity` for vector inputs.

Fields:
$FIELDS
"""
Pigeons.@auto struct VectorizedInterpolatedLP
    """ The enclosed `InterpolatedLogPotential`. """
    enclosed

    """ 
    A version of the reference log potential that implements 
    `LogDensityProblems.logdensity` for vector inputs.
    """
    ref_vectorized

    """ 
    The same as `ref_vectorized` but with the target.
    """
    target_vectorized
end

function (v::VectorizedInterpolatedLP)(x::AbstractVector)
    int_lp = v.enclosed
    beta = int_lp.beta
    return if beta == zero(beta)
        LogDensityProblems.logdensity(v.ref_vectorized, x)
    elseif beta == one(beta)
        LogDensityProblems.logdensity(v.target_vectorized, x)
    else
        Pigeons.interpolate(
            int_lp.path.interpolator, 
            LogDensityProblems.logdensity(v.ref_vectorized, x), 
            LogDensityProblems.logdensity(v.target_vectorized, x), 
            beta
        )
    end
end

# Vectorization of InterpolatedLogPotential
vectorize(int::Pigeons.InterpolatedLogPotential, replica::Pigeons.Replica) =
    VectorizedInterpolatedLP(
        int, 
        vectorize(int.path.ref, replica), 
        vectorize(int.path.target, replica)
    )
