module autoRWMH

using Distributions: UnivariateDistribution, Uniform, Dirac, logpdf, Normal
using DocStringExtensions
using LogDensityProblems
using OnlineStatsBase: value
using Random: AbstractRNG, randn!, randexp
using Statistics: mean

using DynamicPPL: DynamicPPL

using Pigeons
import Pigeons: adapt_preconditioner, Preconditioner, @record_if_requested!,
                get_buffer, build_preconditioner!, MixDiagonalPreconditioner

include("vectorized_logpotentials.jl")
include("StepSizeSelector.jl")

export SimpleRWMH
include("SimpleRWMH.jl")

export HitAndRunSlicer
include("HitAndRunSlicer.jl")

include("explorer_commons.jl")

end # module autoRWMH
