module BayesianScaling

using Dates: Dates, DateTime
using Statistics: Statistics, mean, std, median

using Makie
using DataFrames
using Distributions: Distributions, Distribution, Normal, Uniform, LogNormal, MvLogNormal, MvNormal, Exponential, truncated, logpdf, loglikelihood, convolve
using MLUtils: splitobs
using JSON: JSON
using StatsBase: StatsBase, quantile, sample, mean_and_std, autocor, ecdf
using Random: Random, shuffle!, AbstractRNG
using Optimization: OptimizationProblem, OptimizationFunction, solve
using OptimizationOptimJL: LBFGS
using ADTypes: AutoForwardDiff
using JLD2: jldopen
using DynamicHMC: DynamicHMC, stack_posterior_matrices, mcmc_with_warmup
using ComponentArrays: ComponentArrays, ComponentArray, ComponentVector, FlatAxis
using LogDensityProblemsAD: ADgradient, ADGradientWrapper
using LogDensityProblems: LogDensityProblems, dimension
using Distributions: UnivariateDistribution, support
using TransformedLogDensities: TransformedLogDensity
using TransformVariables: TransformVariables, as, as_real, as_positive_real, as_negative_real
using ProgressBars: ProgressBar
using Format: format
using LogExpFunctions: xexpy
using CategoricalArrays: categorical, levelcode


Statistics.middle(x::ComponentVector, y::ComponentVector) = @. (x + y) / 2

""" Path to exported wandb runs"""
const WANDB_EXPORT_DIR = joinpath(@__DIR__, "..", "..", "..", ".cache", "wandb-export")

""" Petaflop-Day """
const pf_day = 24 * 60 * 60 * 1e15

include("utils.jl")

# Bayesian Modeling of LLM loss curves
include("ppl.jl")
include("scaling.jl")
include("models.jl")

# Analysis of fitted curves
include("checks.jl")
include("plots.jl")
include("figures/summary.jl")

# Planning Tools for LLM Training Campaigns
include("acquire.jl")
include("plan.jl")

include("analysis.jl")

end
