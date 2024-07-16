hoffman_scaling(N, D; A, B, α, β, E) = @. (A / N^α) + (B / D^β) + E
function compute_optimal_model_size(flops; A, α, B, β)
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    return @. G * (flops / 6)^a
end
function compute_optimal_loss(C; A, α, B, β, E)
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    b = @.α / (α + β)
    d = @. C / 6
    return @. E + A * (G * d^a)^(-α) + B * (inv(G) * d^b)^(-β)
end

struct HoffmanScaling{P,X}
    model::BayesModel{P,X}
end
sample_priors(m::HoffmanScaling) = ComponentArray(; sample_priors(m.model)...)
transform_support(m::HoffmanScaling) = transform_support(m.model.priors)

function (m::HoffmanScaling)(θ)
    # Extract parameters
    (; A, B, α, β, E, sigma) = θ

    # Priors
    ℓ = logpdf_prior(m.model.priors, θ)

    # Likelihood
    (; model_size, data_size, loss) = m.model.covariates
    for i in eachindex(loss)
        log_mu = log(hoffman_scaling(model_size[i], data_size[i]; A, B, α, β, E))
        ℓ += loglikelihood(LogNormal(log_mu, sigma), loss[i])
    end
    return ℓ
end

""" Compute the log∘sum∘exp of `x...`"""
lse(x...) = log(sum(exp.(x)))

"""
Loss function for fitting the Hoffman Scaling Model using a Huber loss
"""
function hoffman_huber_loss(θ, p)
    α, β, a, b, e = θ
    (; model_size, data_size, loss) = p
    ℓ = 0.0
    for i in eachindex(loss)
        loss_lse = lse(a - α*log(model_size[i]), b - β*log(data_size[i]), e)
        ℓ += huber_loss(loss_lse - log(loss[i]); δ=1e-3)
    end
    return ℓ
end

huber_loss(x::Real; δ::Real=1e-3) = abs(x) < δ ? 0.5 * x^2 : δ * (abs(x) - 0.5 * δ)

Distributions.quantile(d::UnivariateDistribution, p) = map(x -> quantile(d, x), p)

function fit_model_huber(m::HoffmanScaling; l=0.01)
    priors = m.model.priors
    grid = Iterators.product(
        range(quantile(priors.α, (l, 1-l))...; length=5),
        range(quantile(priors.β, (l, 1-l))...; length=5),
        range(log.(quantile(priors.A, (l, 1-l)))...; length=10),
        range(log.(quantile(priors.B, (l, 1-l)))...; length=10),
        range(log.(quantile(priors.E, (l, 1-l)))...; length=10),
    )

    # Initial grid search
    x0 = argmin(Base.Fix2(hoffman_huber_loss, m.model.covariates), grid) |> splat(vcat)

    # Minimize huber loss
    f = OptimizationFunction(hoffman_huber_loss, AutoForwardDiff())
    x0[3:4] .= 5
    prob = OptimizationProblem(f, x0, m.model.covariates)
    sol = solve(prob, LBFGS(); maxiters=200)
    if sol.retcode != :Success
        @error "Model fitting failed" sol.retcode sol.original sol x0
    end
    θ = ComponentVector(; α=sol.u[1], β=sol.u[2], A=exp(sol.u[3]), B=exp(sol.u[4]), E=exp(sol.u[5]))
    return θ, sol
end

function model_priors(::Type{HoffmanScaling})
    return (;
        A=LogNormal(log(500), 2),
        B=LogNormal(log(500), 2),
        α=truncated(Normal(1, 0.5), 0, Inf),
        β=truncated(Normal(1, 0.5), 0, Inf),
        E=Exponential(1e-3),
        sigma=LogNormal(-1, 2),
    )
end


function HoffmanScaling(df::DataFrame)
    loss = hasproperty(df, :loss) ? df.loss : minimum.(df.val_loss_trace)
    x = ComponentArray(;
        loss,
        model_size=float(df.model_size),
        data_size=float(df.data_size),
    )
    priors = model_priors(HoffmanScaling)
    HoffmanScaling(BayesModel(priors, x))
end

""" Estimate the increase in loss due to the learning rate """
function lr_penalty(lr, N, θ)
    (; lr_0, lr_n, lr_p) = θ
    lr_opt = log(lr_0) - lr_n * log(N)
    return lr_p * (log(lr) - lr_opt)^2
end

struct TrainingProgress{M,N,P,X}
    scaling::M
    penalty::N
    progress::P
    loss::X
    step::X
end

sample_priors(m::TrainingProgress) = ComponentVector(sample_priors(model_priors(TrainingProgress)))
transform_support(m::TrainingProgress) = transform_support(model_priors(TrainingProgress))
function model_priors(::Type{TrainingProgress})
    scaling = model_priors(HoffmanScaling)
    progress = (;
        batch_critical=LogNormal(log(500), 3),
        Sm=LogNormal(log(1), 2),
        γ=LogNormal(0, 1),
    )
    penalty = (;
        lr = (;
            lr_0=LogNormal(log(1e-3), 3),
            lr_n=LogNormal(log(1e-4), 3),
            lr_p=Exponential(1),
        )
    )
    return (; scaling, progress, penalty)
end

function (m::TrainingProgress)(θ)
    # Priors for scaling model
    (; model_size, data_size, eff_batch_size, lr) = m.scaling.covariates
    (; A, B, α, β, E, sigma) = θ.scaling
    (; batch_critical, Sm, γ) = θ.progress

    # Compute Priors
    ℓ = logpdf_prior(m.scaling.priors, θ.scaling)
    ℓ += logpdf_prior(m.progress, θ.progress)
    ℓ += logpdf_prior(m.penalty.lr, θ.penalty.lr)

    # Training Progress Model
    for idx in axes(m.loss, 2)
        # Estimate the minimum loss
        N = model_size[idx]
        D = data_size[idx]
        batch_size = eff_batch_size[idx]
        min_loss = hoffman_scaling(N, D; A, B, α, β, E)

        # LR penalty
        min_loss += lr_penalty(lr[idx], N, θ.penalty.lr)

        # Likelihood for loss curve fits
        steps = m.step[idx]
        loss = m.loss[idx]
        for (s, l) in zip(steps, loss)
            s_eff = s / (1 + batch_critical/batch_size)
            expected_loss = min_loss + Sm / s_eff^γ - Sm
            ℓ += loglikelihood(LogNormal(log(expected_loss), sigma), l)
        end
    end
    return ℓ
end

function TrainingProgress(df::DataFrame)
    scaling_covar = ComponentArray(
        model_size=float(df.model_size),
        data_size=float(df.data_size),
        eff_batch_size=float(df.eff_batch_size),
        lr=float(df.lr),
    )
    (; scaling, progress, penalty) = model_priors(TrainingProgress)
    return TrainingProgress(
        BayesModel(scaling, scaling_covar),
        penalty,
        progress,
        df.val_loss_trace,
        df.rel_step_trace,
    )
end
