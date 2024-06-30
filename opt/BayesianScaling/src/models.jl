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

struct TrainingProgress{M,C,S,X}
    scaling::M
    training_coeffs::C
    training_sigma::S
    loss::X
    step::X
end

function model_priors(::Type{TrainingProgress}, n=1)
    training_coeffs = (;
        a=truncated(Normal(0, 1), 0, Inf),
        b=truncated(Normal(0, 1), 0, Inf),
    )
    scaling = Base.structdiff(model_priors(HoffmanScaling), (; sigma=nothing))
    return (;
        scaling,
        training_sigma=Exponential(1),
        training_coeffs,
    )
end

function (m::TrainingProgress)(θ)
    # Priors for scaling model
    (; model_size, data_size) = m.scaling.covariates
    (; A, B, α, β, E) = θ.scaling
    ℓ = logpdf_prior(m.scaling.priors, θ.scaling)

    # Training Progress Model
    for idx in axes(m.loss, 2)
        # Priors for loss curve fits
        coeffs = @view θ.training_coeffs[:, idx]
        ℓ += logpdf_prior_vec(m.training_coeffs, coeffs)

        # Estimate min loss for the model
        log_min_loss = hoffman_scaling(model_size[idx], data_size[idx]; A, B, α, β, E) |> log

        # Likelihood for loss curve fits
        a, b = coeffs
        σ = θ.training_sigma[idx]
        for tdx in axes(m.step, 1)
            s = m.step[tdx, idx]
            l = m.loss[tdx, idx]
            expected_loss = log_min_loss + a / (s + b) - a / (1 + b)
            ℓ += loglikelihood(LogNormal(expected_loss, σ), l)
        end
    end
    return ℓ
end
function sample_priors(m::TrainingProgress)
    nobs = size(m.loss, 2)
    scaling = sample_priors(m.scaling)
    return ComponentArray(;
        scaling,
        training_coeffs=stack(map(_ -> [sample_priors(m.training_coeffs)...], 1:nobs)),
        training_sigma=rand(m.training_sigma, nobs),
    )
end

function model_priors(m::TrainingProgress)
    return model_priors(TrainingProgress, size(m.loss, 2))
end

function transform_support(m::TrainingProgress)
    np = length(m.training_coeffs)
    nobs = size(m.loss, 2)
    return (;
        scaling=as(transform_support(m.scaling.priors)),
        training_coeffs=as(Matrix, transform_support(first(m.training_coeffs)), np, nobs),
        training_sigma=as(Vector, transform_support(m.training_sigma), nobs),)
end

function TrainingProgress(df::DataFrame)
    scaling_covar = ComponentArray(
        model_size=float(df.model_size),
        data_size=float(df.data_size),
        lr=float(df.lr),
        ff_ratio=float(df.ff_ratio),
        aspect_ratio=float(df.aspect_ratio),
        eff_batch_size=float(df.eff_batch_size),
    )
    (; scaling, training_sigma, training_coeffs) = model_priors(TrainingProgress, nrow(df))
    return TrainingProgress(
        BayesModel(scaling, scaling_covar),
        training_coeffs,
        training_sigma,
        stack(df.val_loss_trace),
        stack(df.rel_step_trace),
    )
end
