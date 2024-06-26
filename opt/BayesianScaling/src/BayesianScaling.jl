module BayesianScaling

using Makie
using DataFrames
using Turing: Turing, sample, @model, @submodel, MLE, NUTS, generated_quantities, Prior, Chains
using ADTypes: AutoZygote
using Distributions: fit, Normal, Uniform, LogNormal, MvLogNormal, Exponential, truncated, logpdf, loglikelihood, convolve
using LogDensityProblems: logdensity
using MLUtils: splitobs
using JSON: JSON
using LinearAlgebra: I, diagm
using Statistics: mean, std, median
using StatsBase: StatsBase, quantile, sample, mean_and_std, autocor, ecdf
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
    α ~ LogNormal(log(0.5), 1)
    β ~ LogNormal(log(0.5), 1)
    E ~ Exponential(1e-3)
    σ² ~ LogNormal(-1, 2)

    # LR Scaling Model
    lr_0 ~ LogNormal(log(1e-3), 2)  # Optimal LR at N = 1
    lr_n ~ Exponential(0.1)         # Scaling of lr with log N
    lr_p ~ Exponential(0.1)         # Penalty term for deviation from lr_eff
    log_lr_eff = @. lr_0 - lr_n * log(model_size)

    # Shape Factors
    ff_ratio_0 ~ LogNormal(log(1), 2)
    ff_ratio_p ~ Exponential(1e-2)
    aspect_ratio_0 ~ LogNormal(log(64), 2)
    aspect_ratio_p ~ Exponential(1e-2)

    # Estimate Loss
    mu = @. (A / (model_size^α)) + (B / (data_size^β)) + E +
            lr_p * (log(lr) - log_lr_eff)^2 +
            ff_ratio_p * logsqdev(ff_ratio, ff_ratio_0) +
            aspect_ratio_p * logsqdev(aspect_ratio, aspect_ratio_0)

    # loss ~ MvLogNormal(log.(mu), σ² * I)
    return mu
end

@model function training_progress(loss, step, model_size, data_size, lr, ff_ratio, aspect_ratio, eff_batch_size)
    # Predict Minimum Loss
    @submodel mu_hoffman = hoffman_scaling(model_size, data_size)
    @submodel penalty_lr = lr_scaling(model_size, lr)
    @submodel eff_step = effective_steps(step, eff_batch_size)
    mu = mu_hoffman .+ penalty_lr
    loss_trace_a = Vector{Float64}(undef, length(loss))
    loss_trace_b = similar(loss_trace_a)
    σ² ~ Exponential(1e-3)
    for i in eachindex(loss)
        loss_trace_a[i] ~ Exponential(1)
        loss_trace_b[i] ~ Exponential(1)
        mu_step = step_log_loss(eff_step[i], mu[i], loss_trace_a[i], loss_trace_b[i])
        loss[i] ~ MvLogNormal(mu_step, I * σ²)
    end
    return [mu mu_hoffman]
end

logistic(x, k=1, x0=0) = 1 / (1 + exp(-k * (x - x0)))
function step_log_loss(rel_step, min_loss, C, D)
    @. log(min_loss) + D/(rel_step + C) - D/(1 + C) # C*(1-rel_step) +
end

@model function effective_steps(steps, eff_batch_size)
    B_crit ~ LogNormal(log(256_000), 3)
    return @. steps / (1 + exp(log(B_crit) - log(eff_batch_size)))
end

@model function lr_scaling(N, lr)
    lr_0 ~ LogNormal(log(1e-3), 2)  # Optimal LR at N = 1
    lr_n ~ Exponential(0.1)         # Scaling of lr with log N
    lr_p ~ Exponential(0.1)         # Penalty term for deviation from lr_eff
    lr_eff = @. lr_0 - lr_n * log(N)
    return @. lr_p * (log(lr) - (lr_0 - lr_n * log(N)))^2
end

@model function hoffman_scaling(model_size, data_size)
    A ~ LogNormal(log(500), 2)
    B ~ LogNormal(log(500), 2)
    α ~ Normal(1, 0.5)
    β ~ Normal(1, 0.5)
    E ~ Exponential(1e-3)
    return @. (A / (model_size^α)) + (B / (data_size^β)) + E
end

@model function ff_penalty(ff_ratio)
    ff_ratio_0 ~ LogNormal(log(1), 2)
    ff_ratio_p ~ Exponential(1e-2)
    return @. ff_ratio_p * logsqdev(ff_ratio, ff_ratio_0)
end

@model function hoffman_scaling(loss, N, D)
    A ~ LogNormal(log(500), 2)
    B ~ LogNormal(log(500), 2)
    α ~ Normal(1, 0.5)
    β ~ Normal(1, 0.5)
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

# hoffman_scaling(N, D; A, α, B, β, E, kwargs...) = @. (A / (N^α)) + (B / (D^β)) + E
function hoffman_scaling(N, D, chains::Chains)
    A = vec(chains[:, :A, :])
    B = vec(chains[:, :B, :])
    α = vec(chains[:, :α, :])
    β = vec(chains[:, :β, :])
    E = vec(chains[:, :E, :])
    return hoffman_scaling(N, D; A, α, B, β, E)
end

function compute_optimal_model(flops; A, α, B, β, kwargs...)
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    return @. G * (flops / 6)^a
end

include("acquire.jl")
include("checks.jl")
include("plan.jl")
include("plots.jl")

end
