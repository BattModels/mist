module BayesianScaling

using Dates: Dates
using Statistics: Statistics
using Random: Random, shuffle!, AbstractRNG
using UUIDs: uuid4

using DataFrames
using Distributions: Distributions, Distribution, Normal, Uniform, LogNormal, MvLogNormal, MvNormal, Exponential, truncated, logpdf, loglikelihood, convolve
using StatsBase: StatsBase, quantile, sample, mean, std, median, mean_and_std, autocor, ecdf, aic, bic, response, coefnames
using StatsModels: StatsModels, apply_schema, schema, FormulaTerm, @formula
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

export @formula


Statistics.middle(x::ComponentVector, y::ComponentVector) = @. (x + y) / 2

""" Petaflop-Day """
const pf_day = 24 * 60 * 60 * 1e15

include("utils.jl")

# Bayesian Regression
include("ppl.jl")
include("checks.jl")

# Neural Scaling Laws
include("scaling.jl")
include("models.jl")

# Planning Tools for LLM Training Campaigns
include("analysis.jl")

end
