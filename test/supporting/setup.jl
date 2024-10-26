# install latest Pigeons
using Pkg
Pkg.add(name="Pigeons", rev="main")

using AutoStep
using BridgeStan
using Distributions
using DynamicPPL
using ForwardDiff
using HypothesisTests
using Pigeons
using Random
using SplittableRandoms
using Test