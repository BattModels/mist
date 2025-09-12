struct HoffmanScaling end

function priors(::Type{HoffmanScaling})
    return (;
        A=LogNormal(log(500), 2.0),
        B=LogNormal(log(500), 2.0),
        α=Uniform(0, 2),
        β=Uniform(0, 2),
        E=LogNormal(log(1e-2), 2),
        sigma=Exponential(0.1),
    )
end

function (::HoffmanScaling)(x, θ)
    return hoffman_scaling(x.model_size, x.data_size; A=θ.A, α=θ.α, B=θ.B, β=θ.β, E=θ.E)
end

function init_model(f::HoffmanScaling, df; priors=priors(HoffmanScaling))
    observations = NamedTuple.(eachrow(df[!, [:model_size, :data_size]]))
    response = df[:, :loss]
    return BayesianRegression(f, priors, response, observations, LogNormal())
end

""" Compute the log∘sum∘exp of `x...`"""
lse(x...) = log(sum(exp.(x)))

huber_loss(x::Real, δ::Real=1e-3) = abs(x) < δ ? 0.5 * x^2 : δ * (abs(x) - 0.5 * δ)

function fit_model_huber(model::BayesianRegression{HoffmanScaling}; p=0.95, n=7, delta=1e-3)
    # Initial grid search over the credible prior
    θ = priors(model)
    grid = Iterators.product(
        range(quantile(θ.α, [p, 1 - p])...; length=n),
        range(quantile(θ.β, [p, 1 - p])...; length=n),
        range(quantile(θ.A, [p, 1 - p])...; length=n) .|> log,
        range(quantile(θ.B, [p, 1 - p])...; length=n) .|> log,
        range(quantile(θ.E, [p, 1 - p])...; length=n) .|> log,
    )
    @assert length(first(grid)) == (dimension(model) - 1)

    function lossfn(θ::Union{Vector,Tuple}, p=nothing)
        loss = zero(eltype(θ))
        α, β, a, b, e = θ
        for (y, x) in zip(response(model), observations(model))
            (; model_size, data_size) = x
            loss_lse = lse(a - α * log(model_size), b - β * log(data_size), e)
            loss += huber_loss(loss_lse - log(y), delta)
        end
        return loss
    end
    x0 = [argmin(lossfn, grid)...]

    # Minimize huber loss
    f = OptimizationFunction(lossfn, AutoForwardDiff())
    prob = OptimizationProblem(f, x0)
    sol = solve(prob, LBFGS(); maxiters=200)
    if sol.retcode != :Success
        @error "Model fitting failed" sol.retcode sol.original sol x0
    end
    θ = (; α=sol.u[1], β=sol.u[2], A=exp(sol.u[3]), B=exp(sol.u[4]), E=exp(sol.u[5]))
    return θ, sol
end

@kwdef struct ShapedScaling
    lr_model_size::Symbol = :model_size
    geometric_penalty::Bool = true
    harmonic_shape_penalty::Bool = true
end

function priors(::Type{ShapedScaling})
    return (;
        scaling=Base.structdiff(priors(HoffmanScaling), NamedTuple{(:sigma,)}),
        lr=(;
            ideal=(;
                a=LogNormal(log(1.6e-4 / 32), 0.5),  # Reference log(LR) for 1024 batch size
                b=Normal(0.5, 0.05),            # Sqrt scaling with effective batch size
                c=Normal(0, 0.1),               # Scaling with model size (assume none)
            ),
            penalty=(Exponential(1.0),),
        ),
        ff_ratio=(LogNormal(log(4), 0.5), Exponential(1.0)),
        kv_size=(LogNormal(log(64), 0.1), Exponential(1.0)),
        aspect_ratio=(LogNormal(log(64), 0.1), Exponential(1.0)),
        sigma=Exponential(0.1),
    )
end

function init_model(f::ShapedScaling, df; priors=priors(HoffmanScaling))
    cols = [f.lr_model_size, :model_size, :data_size, :lr, :ff_ratio, :aspect_ratio, :kv_size, :effective_batch_size]
    unique!(cols)
    observations = NamedTuple.(eachrow(df[!, cols]))
    response = df[:, :loss]
    return BayesianRegression(f, priors, response, observations, LogNormal())
end

function (m::ShapedScaling)(run, θ)
    loss = hoffman_scaling(run.model_size, run.data_size; θ.scaling...)
    P = expected_penalties(m, θ, run).P
    return m.geometric_penalty ? xexpy(loss, P) : loss + P
end

function expected_penalties(m::ShapedScaling, θ, run)
    (; ff_ratio, kv_size, aspect_ratio) = θ
    lr_opt = ideal_lr(m, θ, run)
    P_lr = geometric_penalty(run.lr, lr_opt, θ.lr.penalty...)
    if m.harmonic_shape_penalty
        P_ff = harmonic_penalty(run.ff_ratio, ff_ratio...)
        P_kv = harmonic_penalty(run.kv_size, kv_size...)
        P_a = harmonic_penalty(run.aspect_ratio, aspect_ratio...)
    else
        P_ff = geometric_penalty(run.ff_ratio, ff_ratio...)
        P_kv = geometric_penalty(run.kv_size, kv_size...)
        P_a = geometric_penalty(run.aspect_ratio, aspect_ratio...)
    end
    P = P_lr + P_ff + P_kv + P_a
    return (; P, ff=P_ff, kv=P_kv, aspect=P_a, lr=P_lr)
end

function lr_partial_dependence(model::ShapedScaling, chains::AbstractChains{T}, runs::Vector, y::Vector; p=0.95, factor=:P) where {T}
    residual = Vector{NTuple{3, T}}(undef, length(runs))
    lr_deviance = similar(residual)
    for (i, run) in enumerate(runs)
        resid_ci = credible_interval(p)
        lr_ci = credible_interval(p)
        map(eachslice(chains; dims=(1,2))) do θ
            P = expected_penalties(model, θ, run)
            loss = hoffman_scaling(run.model_size, run.data_size; θ.scaling...)
            P_marginal = P.ff + P.kv + P.aspect
            if model.geometric_penalty
                ŷ =  xexpy(loss, P_marginal)
                resid = y[i] / ŷ
            else
                ŷ =  loss + P_marginal
                resid = y[i] - ŷ
            end
            fit!(resid_ci, resid)

            lr_opt = ideal_lr(model, θ, run)
            lr_dev = model.harmonic_shape_penalty ? run.lr - lr_opt : run.lr / lr_opt
            fit!(lr_ci, lr_dev)
        end
        residual[i] = OnlineStats.value(resid_ci)
        lr_deviance[i] = OnlineStats.value(lr_ci)
    end
    return lr_deviance, residual
end

function ideal_lr(m::ShapedScaling, θ, run)
    (; a, b, c) = θ.lr.ideal
    N = convert(Float64, run[m.lr_model_size])
    return exp(log(a) + b * log(run.effective_batch_size) + c * log(N))
end

function ideal_lr(m::ShapedScaling, chains::AbstractChains, N::Real, B::Real; p=0.95)
    lr = credible_interval(p)
    for J in CartesianIndices(axes(chains)[1:2])
        θ = chains[J.I..., :]
        fit!(lr, ideal_lr(m, θ, (; m.lr_model_size => N, :effective_batch_size => B)))
    end
    return OnlineStats.value(lr)
end

function ideal_lr(model::ShapedScaling, chains::AbstractChains, N, B; p=0.95)
    mu = similar(chains, length(N), length(B))
    lower = similar(mu)
    upper = similar(mu)
    for J in eachindex(IndexCartesian(), mu)
        mu[J], lower[J], upper[J] = ideal_lr(model, chains, N[J[1]], B[J[2]]; p)
    end
    return mu, lower, upper
end
