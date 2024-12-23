hoffman_scaling(N, D; A, B, α, β, E) = @. (A / N^α) + (B / D^β) + E
function compute_optimal_model_size(flops; A, α, B, β)
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    return @. G * (flops / 6)^a
end
function compute_optimal_data_size(flops; A, α, B, β)
    G_inv = @. ((α * A) / (β * B))^(-1 / (α + β))
    b = @. α / (α + β)
    return @. G_inv * (flops / 6)^b
end
function compute_optimal_loss(C; A, α, B, β, E)
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    b = @. α / (α + β)
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
        loss_lse = lse(a - α * log(model_size[i]), b - β * log(data_size[i]), e)
        ℓ += huber_loss(loss_lse - log(loss[i]); δ=1e-3)
    end
    return ℓ
end

huber_loss(x::Real; δ::Real=1e-3) = abs(x) < δ ? 0.5 * x^2 : δ * (abs(x) - 0.5 * δ)

Distributions.quantile(d::UnivariateDistribution, p) = map(x -> quantile(d, x), p)

function fit_model_huber(m::HoffmanScaling; l=0.01)
    priors = m.model.priors
    grid = Iterators.product(
        range(quantile(priors.α, (l, 1 - l))...; length=5),
        range(quantile(priors.β, (l, 1 - l))...; length=5),
        range(log.(quantile(priors.A, (l, 1 - l)))...; length=10),
        range(log.(quantile(priors.B, (l, 1 - l)))...; length=10),
        range(log.(quantile(priors.E, (l, 1 - l)))...; length=10),
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
        A=LogNormal(log(500), 2.0),
        B=LogNormal(log(500), 2.0),
        α=Uniform(0, 2),
        β=Uniform(0, 2),
        E=LogNormal(log(1e-3), 1),
        sigma=LogNormal(-1, 2.0),
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


struct ShapedScaling{P,R}
    priors::P
    runs::Vector{R}
end
sample_priors(m::ShapedScaling) = ComponentVector(sample_priors(m.priors))
transform_support(m::ShapedScaling) = transform_support(m.priors)

function model_priors(::Type{ShapedScaling})
    return (;
        scaling=Base.structdiff(model_priors(HoffmanScaling), NamedTuple{(:sigma,)}),
        lr=(;
            ideal=(;
                a=Normal(log(3.9e-5), 1),     # Reference log(LR)
                b=Normal(0.5, 0.1),           # Scaling with effective batch size
                c=Normal(0, 0.5),             # Scaling with model size
            ),
            penalty=(Exponential(1.0),),
        ),
        # beta=(;
        #     ideal=(;
        #         beta1=Uniform(0, 1),
        #         beta2=Uniform(0, 1),
        #     ),
        #     penalty=(Exponential(1.0),),
        # ),
        ff_ratio=(LogNormal(log(4), log(2)), Exponential(1.0)),
        kv_size=(LogNormal(log(64), 1), Exponential(1.0)),
        aspect_ratio=(LogNormal(log(64), 2), Exponential(1.0)),
        sigma=Exponential(1.0),
    )
end

function ShapedScaling(df::DataFrame)
    cols = [:loss, :model_size, :data_size, :lr, :ff_ratio, :aspect_ratio, :kv_size, :effective_batch_size, :beta1, :beta2]
    runs = map(NamedTuple, eachrow(df[!, cols]))
    priors = model_priors(ShapedScaling)
    return ShapedScaling(priors, runs)
end

StatsBase.response(m::ShapedScaling) = map(run -> run.loss, m.runs)

function (m::ShapedScaling)(θ)
    (; scaling, lr, ff_ratio, aspect_ratio, kv_size, sigma) = θ

    # Priors
    ℓ = logpdf_prior(m.priors.scaling, scaling)
    ℓ += logpdf_prior(m.priors.lr, lr)
    # ℓ += logpdf_prior(m.priors.beta, beta)
    ℓ += logpdf_prior(m.priors.ff_ratio, ff_ratio)
    ℓ += logpdf_prior(m.priors.aspect_ratio, aspect_ratio)
    ℓ += logpdf_prior(m.priors.kv_size, kv_size)

    # Likelihood
    ℓ += sum(m.runs) do run
        min_loss = hoffman_scaling(run.model_size, run.data_size; scaling...) |> log
        min_loss += lamb_penalty(run.lr, run.model_size, run.effective_batch_size; lr...)
        # min_loss += beta_penalty(run.beta1, run.beta2, run.effective_batch_size; beta...)
        min_loss += harmonic_penalty(run.ff_ratio, ff_ratio...)
        min_loss += harmonic_penalty(run.kv_size, kv_size...)
        min_loss += harmonic_penalty(run.aspect_ratio, aspect_ratio...)
        return loglikelihood(LogNormal(min_loss, sigma), run.loss)
    end
    return ℓ
end

function sample_response(m::ShapedScaling, chains; nsamples=1_000)
    nruns = length(m.runs)
    y_hat = Array{Float64}(undef, nruns, nsamples)
    indices = CartesianIndices((axes(chains, 1), axes(chains, 2)))
    for (idx, run) in enumerate(m.runs)
        for sdx in 1:nsamples
            I = rand(indices).I
            θ = chains[I..., :]
            y_hat[idx, sdx] = expected_loss(m, θ, run)
        end
    end
    return y_hat
end

function sample_penalty(m::ShapedScaling, chains; nsamples=1_000)
    nruns = length(m.runs)
    P = Array{Float64}(undef, nruns, nsamples, 4)
    indices = CartesianIndices((axes(chains, 1), axes(chains, 2)))
    for (idx, run) in enumerate(m.runs)
        for sdx in 1:nsamples
            I = rand(indices).I
            θ = chains[I..., :]
            p_lr = lamb_penalty(run.lr, run.model_size, run.effective_batch_size; θ[:lr]...)
            p_ff = harmonic_penalty(run.ff_ratio, θ[:ff_ratio]...)
            p_kv = harmonic_penalty(run.kv_size, θ[:kv_size]...)
            p_aspect = harmonic_penalty(run.aspect_ratio, θ[:aspect_ratio]...)
            P[idx, sdx, :] .= [p_lr, p_ff, p_kv, p_aspect]
        end
    end
    return P
end


function expected_loss(::ShapedScaling, θ, run::NamedTuple)
    (; scaling, lr, ff_ratio, aspect_ratio, kv_size, sigma) = θ
    min_loss = hoffman_scaling(run.model_size, run.data_size; scaling...) |> log
    min_loss += lamb_penalty(run.lr, run.model_size, run.effective_batch_size; lr...)
    # min_loss += beta_penalty(run.beta1, run.beta2, run.effective_batch_size; beta...)
    min_loss += harmonic_penalty(run.ff_ratio, ff_ratio...)
    min_loss += harmonic_penalty(run.kv_size, kv_size...)
    min_loss += harmonic_penalty(run.aspect_ratio, aspect_ratio...)
    return exp(min_loss)
end

function beta_penalty(beta1, beta2, eff_batch_size; ideal, penalty)
    (; beta1, beta2) = ideal
    κ = eff_batch_size / 1024
    beta_clamp(x) = clamp(x, 0, 1)
    β₁ = 1 - κ * (1 - beta1) |> beta_clamp
    β₂ = 1 - κ * (1 - beta2) |> beta_clamp
    p1 = geoharmonic_penalty(beta1, β₁, penalty...)
    p2 = geoharmonic_penalty(beta2, β₂, penalty...)
    return p1 + p2
end


geoharmonic_penalty(args...) = 1 + harmonic_penalty(args...)
function harmonic_penalty(x, x0::T, penalty::T...) where {T<:Real}
    δ = (T(x) - x0)^2
    l = first(penalty) * δ
    for p in Base.tail(penalty)
        δ *= δ
        l += p * δ
    end
    return l
end
function geometric_penalty(x, x0::T, penalty::T...) where {T<:Real}
    δ = (log(T(x)) - log(x0))^2
    l = first(penalty) * δ
    for p in Base.tail(penalty)
        δ *= δ
        l += p * δ
    end
    return l
end

function lamb_penalty(lr, model_size, effective_batch_size; ideal, penalty)
    (; a, b, c) = ideal
    lr_0 = a + b * log(effective_batch_size) + c * log(model_size) |> exp
    return geometric_penalty(lr, lr_0, penalty...)
end

function ideal_lr(model::ShapedScaling, chains::AbstractArray{T,3}, df::DataFrame) where {T}
    lr = Array{T}(undef, nrow(df), size(chains)[1:2]...)
    indices = CartesianIndices(axes(chains)[1:2])
    for (rdx, row) in enumerate(eachrow(df))
        for sdx in indices
            I = sdx.I
            lr[rdx, I...] = ideal_lr(model, chains[I..., :], row.model_size, row.effective_batch_size)
        end
    end
    return lr
end

function ideal_lr(model::ShapedScaling, chains::AbstractArray{T,3}; model_size, effective_batch_size) where {T}
    lr = Array{T}(undef, size(chains)[1:2]...)
    for sdx in CartesianIndices(axes(lr))
        lr[sdx] = ideal_lr(model, chains[sdx.I..., :], model_size, effective_batch_size)
    end
    return lr
end

function ideal_lr(::ShapedScaling, θ::ComponentVector, model_size, effective_batch_size)
    (; a, b, c) = θ.lr.ideal
    return exp(a + b * log(effective_batch_size) + c * log(model_size))
end

function ideal_lr!(lr::AbstractArray{T,2}, m, chains::AbstractArray{T,3}, args...) where {T}
    for idx in CartesianIndices(axes(chains)[1:2])
        lr[idx.I...] = ideal_lr(m, chains[idx.I..., :], args...)
    end
    return lr
end

function ideal_lr_map(m::ShapedScaling, chains::AbstractArray{T,3}, model_size, effective_batch_size; p=[0.025, 0.5, 0.975]) where {T}
    lr_map = Array{T}(undef, length(p), length(model_size), length(effective_batch_size))
    lr = Array{T}(undef, size(chains)[1:2]...)
    for (mdx, ms) in enumerate(model_size)
        for (edx, ebs) in enumerate(effective_batch_size)
            ideal_lr!(lr, m, chains, ms, ebs)
            lr_map[:, mdx, edx] .= quantile(lr, p)
        end
    end
    return lr_map
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
    progress = (;
        batch_critical=LogNormal(log(500), 3),
        Sm=LogNormal(log(1), 2),
        γ=LogNormal(0, 1),
    )
    penalty = (;
        lr=(;
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
            s_eff = s / (1 + batch_critical / batch_size)
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
