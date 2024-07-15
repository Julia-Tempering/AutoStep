module autoRWMH

using Distributions: UnivariateDistribution, Uniform, Dirac, logpdf
using DocStringExtensions
using LogDensityProblemsAD: ADgradient
using OnlineStatsBase: value
using Random: AbstractRNG, randn!, randexp
using Statistics: mean

using Pigeons
import Pigeons: adapt_preconditioner, Preconditioner, @record_if_requested!,
                get_buffer, build_preconditioner!, hamiltonian_dynamics!,
                grow_step_size, shrink_step_size, MixDiagonalPreconditioner

include("StepSizeSelector.jl")

export SimpleRWMH
include("SimpleRWMH.jl")

end # module autoRWMH
