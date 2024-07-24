# install latest Pigeons
using Pkg
Pkg.add(name="Pigeons", rev="main")

using autoRWMH
using BridgeStan
using Distributions
using DynamicPPL
using HypothesisTests
using Pigeons
using Random
using SplittableRandoms
using Test
