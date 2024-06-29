module BayesianScaling

using Makie
using DataFrames
using Distributions: Distribution, Normal, Uniform, LogNormal, MvLogNormal, MvNormal, Exponential, truncated, logpdf, loglikelihood, convolve
using MLUtils: splitobs
using JSON: JSON
using Statistics: mean, std, median
using StatsBase: StatsBase, quantile, sample, mean_and_std, autocor, ecdf
using Random: Random, shuffle!, AbstractRNG

using JLD2: jldopen
using DynamicHMC: DynamicHMC
using ComponentArrays: ComponentArrays, ComponentArray, ComponentVector
using LogDensityProblemsAD: ADgradient, ADGradientWrapper
using LogDensityProblems: LogDensityProblems, dimension
using Distributions: UnivariateDistribution, support
using TransformedLogDensities: TransformedLogDensity
using TransformVariables: TransformVariables, as, as_real, as_positive_real, as_negative_real


""" Petaflop-Day """
const pf_day = 24 * 60 * 60 * 1e15

include("utils.jl")

# Bayesian Modeling of LLM loss curves
include("ppl.jl")
include("models.jl")

# Analysis of fitted curves
include("checks.jl")
include("plots.jl")

# Planning Tools for LLM Training Campaigns
include("acquire.jl")
include("plan.jl")

end
