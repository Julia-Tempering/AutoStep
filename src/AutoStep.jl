module AutoStep

using Distributions: Distributions, UnivariateDistribution, Uniform, Dirac, logpdf, Normal
using DocStringExtensions
using LogDensityProblems
using LogDensityProblemsAD: ADgradient
using Random: AbstractRNG, randn!, randexp
using Statistics: mean, median

using DynamicPPL: DynamicPPL

using Pigeons
import Pigeons: adapt_preconditioner, Preconditioner, @record_if_requested!,
                log_joint, hamiltonian_dynamics!, grow_step_size, shrink_step_size, 
                get_buffer, build_preconditioner!, MixDiagonalPreconditioner

include("vectorized_logpotentials.jl")
include("IntegrationTime.jl")
include("StepSizeSelector.jl")
include("StepJitter.jl")

export SimpleRWMH
include("SimpleRWMH.jl")

export HitAndRunSlicer
include("HitAndRunSlicer.jl")

export SimpleAHMC
include("SimpleAHMC.jl")

include("explorer_commons.jl")

end # module AutoStep
