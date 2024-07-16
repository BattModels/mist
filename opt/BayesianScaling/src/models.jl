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
