module BayesianScaling

using Makie
using DataFrames
using Turing: Turing, sample, @model, MLE, NUTS, generated_quantities, Prior, Chains
using ADTypes: AutoZygote
using Distributions: fit, Normal, Uniform, LogNormal, MvLogNormal, Exponential, truncated, logpdf, loglikelihood, convolve
using LogDensityProblems: logdensity
using MLUtils: splitobs
using JSON: JSON
using LinearAlgebra: I, diagm
using Statistics: mean, std, median
using StatsBase: quantile, sample, mean_and_std, autocor, ecdf
using Random: shuffle!
using MCMCChainsStorage: MCMCChainsStorage

export Chains

""" Petaflop-Day """
const pf_day = 24 * 60 * 60 * 1e15

include("utils.jl")

@model function bayes_llm_scaling_model(loss, model_size, data_size, lr, ff_ratio, aspect_ratio)
    # Base LLM Scaling model from: Hoffmann, J. et al. 2022.
    # Training Compute-Optimal Large Language Models. arXiv.
    A ~ LogNormal(log(500), 2)
    B ~ LogNormal(log(500), 2)
    α ~ truncated(Normal(0.1, 0.5), 0, 2)
    β ~ truncated(Normal(0.1, 0.5), 0, 2)
    E ~ Exponential(1e-3)
    σ² ~ LogNormal(-1, 2)

    # LR Scaling Model
    lr_0 ~ LogNormal(log(5e-5), 2)  # Optimal LR at N = 1
    lr_n ~ LogNormal(log(1e-2), 4)  # Scaling of lr with log N
    lr_p ~ Exponential(0.1)    # Penalty term for deviation from lr_eff
    log_lr_eff = @. lr_0 - lr_n * log(model_size)

    # Shape Factors
    ff_ratio_0 ~ LogNormal(log(1), 2)   # Ideal ff_ratio
    ff_ratio_p ~ Exponential(0.1)        # Penalty term for deviation
    aspect_ratio_0 ~ LogNormal(log(64), 2)
    aspect_ratio_p ~ Exponential(0.1)

    # Estimate Loss
    mu = @. (A / (model_size^α)) + (B / (data_size^β)) + E +
            lr_p * (log(lr) - log_lr_eff)^2 +
            ff_ratio_p * logsqdev(ff_ratio, ff_ratio_0) +
            aspect_ratio_p * logsqdev(aspect_ratio, aspect_ratio_0)

    loss ~ MvLogNormal(log.(mu), σ² * I)
    return mu
end

@model function hoffman_scaling(loss, N, D)
    A ~ LogNormal(log(500), 2)
    B ~ LogNormal(log(500), 2)
    α ~ truncated(Normal(0.1, 0.5), 0, 2)
    β ~ truncated(Normal(0.1, 0.5), 0, 2)
    E ~ LogNormal(-2, 5)
    σ² ~ LogNormal(-1, 2)

    mu = hoffman_scaling.(N, D; A, α, B, β, E)
    if loss isa Missing
        loss ~ LogNormal(log(mu), σ²)
    else
        loss ~ MvLogNormal(log.(mu), σ² * I)
    end
    return mu
end

hoffman_scaling(N, D; A, α, B, β, E, kwargs...) = @. (A / (N^α)) + (B / (D^β)) + E

function compute_optimal_model(flops; A, α, B, β, kwargs...)
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    return @. G * (flops / 6)^a
end

include("plan.jl")
include("plots.jl")

end
