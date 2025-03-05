module BayesianScaling

using Statistics: Statistics, mean, std, median

using Makie
using DataFrames
using Distributions: Distributions, Distribution, Normal, Uniform, LogNormal, MvLogNormal, MvNormal, Exponential, truncated, logpdf, loglikelihood, convolve
using StatsBase: StatsBase, quantile, sample, mean_and_std, autocor, ecdf, aic, bic, logpdf, response
using Random: Random, shuffle!, AbstractRNG
using Optimization: OptimizationProblem, OptimizationFunction, LBFGS, solve
using OnlineStats: OnlineStats, OnlineStat, KHist, fit!
using ADTypes: AutoForwardDiff
using JLD2: jldopen, jldsave
using DynamicHMC: DynamicHMC, stack_posterior_matrices, mcmc_with_warmup
using ComponentArrays: ComponentArrays, ComponentArray, ComponentVector, FlatAxis
using LogDensityProblemsAD: ADgradient, ADGradientWrapper
using LogDensityProblems: LogDensityProblems, dimension, logdensity
using Distributions: UnivariateDistribution, support
using TransformedLogDensities: TransformedLogDensity
using TransformVariables: TransformVariables, as, as_real, as_positive_real, as_negative_real
using LogExpFunctions: LogExpFunctions, xexpy
using MCMCDiagnosticTools: ess_rhat


Statistics.middle(x::ComponentVector, y::ComponentVector) = @. (x + y) / 2

""" Petaflop-Day """
const pf_day = 24 * 60 * 60 * 1e15

include("utils.jl")

# Bayesian Regression
include("ppl.jl")

# Neural Scaling Laws
include("scaling.jl")
include("models.jl")

# Analysis of fitted curves
# include("checks.jl")
# include("plots.jl")
# include("figures/summary.jl")

# Planning Tools for LLM Training Campaigns
# include("acquire.jl")
# include("plan.jl")

# include("analysis.jl")

end
