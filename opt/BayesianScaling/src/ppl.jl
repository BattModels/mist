const AbstractChains{T} = AbstractArray{T,3} where {T}

const Fitted{M,T} = Tuple{M, AbstractChains{T}} where {M,T}

struct BayesModel{P,X}
    priors::P
    covariates::X
end


"""
    priors(model)

The prior distributions `p(θ)` of the `model`
"""
function priors end
priors(::T) where {T} = priors(T)

"""
    observations(model)

Features of the `model`
"""
function observations end

"""
    predict(model, θ, x)

Return a new prediction for `x` under the parameters `θ`
"""
function predict end

"""
    deviance_logdensity(model, θ, y, x)

Return the likelihood of `y` given `x` and `θ` """
function deviance_logdensity end


struct BayesianRegression{M,P,O,R,D}
    formula::M
    priors::P
    response::Vector{R}
    observations::Vector{O}
    dist::D
end

Base.eltype(m::BayesianRegression) = eltype(m.response)
priors(m::BayesianRegression) = m.priors
StatsBase.response(m::BayesianRegression) = m.response
observations(m::BayesianRegression) = m.observations
predict(m::BayesianRegression, θ, x) = m.formula(x, θ)
deviance_logdensity(m::BayesianRegression, y, ŷ; θ) = loglike_obs(m.dist, y, ŷ, dispersion(m, θ))
dispersion(::BayesianRegression, θ) = θ.sigma

function Base.show(io::IO, ::MIME"text/plain", m::BayesianRegression)
    n = length(m.response)
    println(io, "BayesianRegression")
    println(io, "  formula: ", m.formula)
    println(io, "  $n observations of ", keys(first(m.observations)))
    println(io, "  priors: ", priors(m))
end

LogDensityProblems.dimension(m::BayesianRegression) = TransformVariables.dimension(transform_support(m))

transform_support(m::BayesianRegression) = transform_support(priors(m))

LogDensityProblems.logdensity(m::BayesianRegression, θ) = logpdf(:joint, m, θ)

Distributions.logpdf(s::Symbol, m::BayesianRegression, θ) = logpdf(Val(s), m, θ)

""" log-density of the parameters given the prior: `log p(θ)` """
Distributions.logpdf(::Val{:prior}, m::BayesianRegression, θ) = logpdf_prior(priors(m), θ)

""" log-density of the parameters given the prior and likelihood: `log p(x, θ) = log p(θ | x) + log p(θ)` """
Distributions.logpdf(::Val{:joint}, m::BayesianRegression, θ) = logpdf(:prior, m, θ) + logpdf(:likelihood, m, θ)

""" Loglikelihood of the parameters given the data: `log p(θ | x)` """
function Distributions.logpdf(::Val{:likelihood}, m::BayesianRegression, θ)
    ϕ = dispersion(m, θ)
    D = m.dist
    ℓ = zero(eltype(m))
    for i in eachindex(m.response)
        y = m.response[i]
        x = m.observations[i]
        ŷ = predict(m, θ, x)
        ℓ += loglike_obs(D, y, ŷ, ϕ)
    end
    return ℓ
end


"""
    logpdf_prior(priors::NamedTuple, θ)

Compute the logdensity of the priors `priors` at `θ`, expects `θ` to define
`getproperty` for each key in `priors` (i.e. `θ` is a `NamedTuple` or `ComponentVector`).
"""
@generated function logpdf_prior(priors::NamedTuple{names}, θ) where {names}
    exprs = []
    for name in names
        push!(exprs, :(logpdf_prior(priors.$name, θ.$name)))
    end
    return Expr(:call, +, exprs...)
end

function logpdf_prior(priors::Tuple{Vararg{<:Distribution}}, θ)
    return sum(zip(priors, θ)) do (p, x)
        logpdf_prior(p, x)
    end
end

logpdf_prior(prior::Distribution, θ::Number) = logpdf(prior, θ)

""" Loglikelihood of an observation `y ~ D(ŷ, ϕ)` """
loglike_obs(::D, y, ŷ, ϕ) where {D <: UnivariateDistribution} = logpdf(D(ŷ, ϕ), y)

"""
    logpdf_prior_vec(priors::NamedTuple, θ::AbstractVector)

Compute the logdensity of the priors `priors` at `θ`, expects a 1:1 mapping of elements in `θ` to `priors`.
(i.e. each distribution in `priors` is univariate)
"""
@generated function logpdf_prior_vec(priors::NamedTuple{names}, θ::AbstractVector) where {names}
    exprs = []
    for (idx, name) in enumerate(names)
        push!(exprs, :(logpdf(priors.$name, θ[$idx])))
    end
    return Expr(:call, +, exprs...)
end

sample_priors(m::BayesianRegression) = sample_priors(priors(m))
sample_priors(m::NamedTuple) = NamedTuple{keys(m)}(map(sample_priors, values(m)))
sample_priors(m::Tuple{Vararg{<:Distribution}}) = map(sample_priors, m)
sample_priors(m::Distribution) = rand(m)

"""
    lb, ub = credible_prior(model, p=0.95)

Return the lower and upper bounds of the credible interval for the priors of `model`
"""
function credible_prior(model, p::Real=0.95)
    p = (1 - p) / 2
    prior = priors(model)
    lb = map(dist -> quantile(dist, p), values(prior))
    ub = map(dist -> quantile(dist, 1 - p), values(prior))
    return lb, ub
end


"""
    t = transform_support(prior)

Construct a transform using [TransfromVariables](https://github.com/tpapp/TransformVariables.jl) for mapping ℝ to the support of `prior`
`prior` can be a `NamedTuple` of distributions or a single distribution.
"""
function transform_support(d::UnivariateDistribution)
    (; lb, ub) = support(d)
    if lb == -Inf && ub == Inf
        return as_real
    elseif lb == -Inf
        return as_negative_real
    elseif ub == Inf
        return as_positive_real
    end
    return as(Real, lb, ub)
end
transform_support(p::NamedTuple) = as(map(transform_support, p))
transform_support(p::Tuple{Vararg{<:Distribution}}) = as(map(transform_support, p))
TransformVariables.as(t::NamedTuple) = as(map(TransformVariables.as, t))

function init_logdensity_model(m, adtype=:Enzyme)
    t = transform_support(m)
    p = TransformedLogDensity(t, Base.Fix1(LogDensityProblems.logdensity, m))
    ∇P = ADgradient(adtype, p)
    return ∇P
end

"""
    y = transform_samples(t, x::Matrix)

Apply the transform `t` to each column of `x`
"""
function transform_samples(t::TransformVariables.AbstractTransform, x::Matrix{T}) where {T<:Real}
    y = similar(x)
    for idx in axes(x, 2)
        ys = @view y[:, idx]
        xs = @view x[:, idx]
        transform!(ys, t, xs)
    end
    return y
end

function transform!(y::AbstractVector, tt::TransformVariables.TransformTuple, x::AbstractVector)
    (; transformations) = tt
    @assert TransformVariables.dimension(tt) == length(y) == length(x)
    index = firstindex(y)
    for t in transformations
        d = TransformVariables.dimension(t)
        if d == 1
            if t isa TransformVariables.ScalarTransform
                y[index] = TransformVariables.transform(t, x[index])
            else
                y[index] = TransformVariables.transform(t, x[index:index]) |> first
            end
        else
            si = range(index; length=d)
            ys = view(y, si)
            xs = view(x, si)
            transform!(ys, t, xs)
        end
        index += d
    end
    return y
end
transform!(y::AbstractVector, t::TransformVariables.AbstractTransform, x::AbstractVector) = copyto!(y, TransformVariables.transform(t, x))

function transfrom_axis(tt::TransformVariables.TransformTuple{<:NamedTuple})
    ax_tt = []
    index = 1
    for (k, t) in pairs(tt.transformations)
        ax = transfrom_axis(t)
        n = TransformVariables.dimension(t)
        if ax isa Union{ComponentArrays.ShapedAxis,ComponentArrays.Axis}
            ax = ComponentArrays.ViewAxis(range(index; length=n), ax)
        else
            ax = ComponentArrays.reindex(ax, index - 1)
        end
        index += TransformVariables.dimension(t)
        push!(ax_tt, k => ax)
    end
    return ComponentArrays.Axis(NamedTuple(ax_tt))
end

function transfrom_axis(t::TransformVariables.AbstractTransform)
    if TransformVariables.dimension(t) == 1
        return ComponentArrays.ViewAxis(1)
    end
    return ComponentArrays.ViewAxis(range(1; length=TransformVariables.dimension(t)))
end

function transfrom_axis(t::TransformVariables.ArrayTransformation)
    if length(t.dims) == 1
        return ComponentArrays.ViewAxis(1:t.dims[1])
    else
        return ComponentArrays.ShapedAxis(t.dims)
    end
end

function sample_chains(ℓ; nchains=15, draws=1_000, adtype=:Enzyme)
    model = init_logdensity_model(ℓ, adtype)
    reporter = DynamicHMC.NoProgressReport()
    posterior = Vector{Matrix{Float64}}(undef, nchains)
    tt = model.ℓ.transformation
    Threads.@threads :dynamic for i in 1:nchains
        y_raw = mcmc_with_warmup(Random.default_rng(), model, draws; reporter).posterior_matrix
        posterior[i] = transform_samples(tt, y_raw)
    end
    y = stack(posterior'; dims=2)
    @assert size(y) == (draws, nchains, dimension(model))

    # Add ComponentVectors to the chains
    ax = transfrom_axis(model.ℓ.transformation)
    y = ComponentArray(y, FlatAxis(), FlatAxis(), ax)
    return y
end

"""
    save_results(model, chains, raw_chains; outdir)

Save the model and sample chains to the output directory
"""
function save_results(model, chains, raw_chains; outdir=joinpath(pkgdir(@__MODULE__), "out"))
    model_name = string(uuid4())
    outdir = joinpath(outdir, model_name)
    mkpath(outdir)
    metadata = (;
        git=readchomp(`git describe --all --long --dirty`),
        timestamp=string(Dates.now()),
    )
    jldsave(joinpath(outdir, "chains.jld2"); model, chains, raw_chains, metadata)
    return outdir
end

function mapchains(f, chains::AbstractArray{<:Real,3}, x::Vector)
    out = similar(chains, size(chains, 1), size(chains, 2), length(x))
    for I in CartesianIndices(axes(chains)[1:2])
        for (rdx, x) in enumerate(x)
            θ = @view chains[I.I..., :]
            out[I, rdx] = f(x, θ)
        end
    end
    return out
end

"""
    chain = subsample(chains::AbstractArray{T,3}, n::Integer)

Draw `n` samples from each chain in `chains` of size `(draws, chains, parameters)`
"""
function subsample(chains::AbstractArray{T,3}, n::Integer) where {T}
    s = similar(chains, n, 1, size(chains, 3))
    draws = CartesianIndices((axes(chains, 1), axes(chains, 2)))
    for cdx in 1:n
        i = rand(draws)
        θ = @view chains[i.I..., :]
        copyto!(selectdim(s, 1, cdx), θ)
    end
    if chains isa ComponentArray
        s = ComponentArray(s, chains.axes...)
    end
    return s
end

function subsample(chains::AbstractArray{T,3}, p::AbstractFloat=0.2) where {T}
    @assert 0 < p <= 1 lazy"Expected p ∈ (0, 1], got $p"
    ns = size(chains, 1) * size(chains, 2)
    n = ceil(Int, ns * p)
    return subsample(chains, n)
end


"""
    μ, l, u = credible_interval(chains; p=0.95)

Return the expectation of `chains` and the `p`-percentile credible interval
"""
function credible_interval(chains::AbstractArray{T,3}; p=0.95) where {T}
    p = (1 - p) / 2
    μ = vec(mean(chains; dims=(1, 2)))
    l = similar(μ)
    u = similar(l)
    for (idx, chains) in enumerate(eachslice(chains; dims=3))
        l[idx], u[idx] = quantile(chains, (p, 1 - p))
    end
    return μ, l, u
end
function credible_interval(chains::AbstractMatrix{T}; p::AbstractFloat=0.95) where {T}
    p = (1 - p) / 2
    return mean(chains), quantile(vec(chains), (p, 1 - p))...
end

function credible_interval(p=0.90)
    p = (1 - p) / 2
    return OnlineStats.Series(
        OnlineStats.Mean(),
        OnlineStats.P2Quantile(p),
        OnlineStats.P2Quantile(1 - p),
    )
end

"""
Compute `y/exp(E[log(ŷ) - log(P)])` where `y` is the response, `ŷ` is the expected response and `P` is a penalty term
It's expected that `ŷ = f(x...) × P`
"""
function penalty_residual(y::Vector{T}, y_hat::AbstractArray{T,3}, penalty::AbstractArray{T,3}) where {T<:Real}
    mu = similar(y_hat)
    for I in eachindex(IndexCartesian(), y_hat)
        mu[I] = log(y_hat[I]) - penalty[I]
    end
    mu = vec(mean(mu; dims=(1, 2)))
    return xexpy.(y, -mu)
end

"""
The maximum a posteriori estimate of the parameters from the sampled `chains`
"""
function maximum_posterior_estimate(model::BayesianRegression, chains::AbstractChains)
    ℓ = θ -> logpdf(:likelihood, model, θ)
    return argmax(ℓ, eachslice(chains; dims=(1, 2))) |> copy
end

function maximum_posterior_estimate(model::BayesianRegression; p=0.95, n=7, adtype=:Enzyme, alg=LBFGS(), kwargs...)
    L = init_logdensity_model(model, adtype)
    f = OptimizationFunction(
        (u, p) -> -LogDensityProblems.logdensity(L, u);
        grad=(G, u, p) -> G .= -last(LogDensityProblems.logdensity_and_gradient(L, u))
    )

    # Initial grid search over the credible prior
    lb, ub = credible_prior(model, p)
    grid = Iterators.product(map(int -> range(int...; length=n), zip(lb, ub))...)
    θ_init = argmin(grid) do x
        θ = NamedTuple{keys(model.priors)}(x)
        return logpdf(:likelihood, model, θ)
    end
    θ_init = NamedTuple{keys(model.priors)}(θ_init)
    x0 = TransformVariables.inverse(L.ℓ.transformation, θ_init)

    # Maximum the joint posterior
    sol = solve(OptimizationProblem(f, x0), alg; kwargs...)
    x = TransformVariables.transform(L.ℓ.transformation, sol.u)
    return x, sol
end

function StatsBase.aic(model::BayesianRegression, chains::AbstractChains)
    ℓ_mle = maximum(θ -> logpdf(:likilhood, model, θ), eachslice(chains; dims=(1, 2)))
    k = dimension(model)
    return 2 * (k - ℓ_mle)
end

function StatsBase.bic(model::BayesianRegression, chains::AbstractChains)
    ℓ_mle = maximum(θ -> logpdf(:likelihood, model, θ), eachslice(chains; dims=(1, 2)))
    k = dimension(model)
    n = length(observations(model))
    return 2 * (k * log(n) - ℓ_mle)
end

"""
Deviance Information Criterion of a Bayesian model
"""
function dic(model, chains::AbstractChains{T}) where {T}
    slices = eachslice(chains; dims=(1, 2))
    θ_bayes = mean(slices)
    lpd = logpdf(:likelihood, model, θ_bayes)
    elpd = mean(θ -> logpdf(:likelihood, model, θ), slices)
    p_dic = 2 * (lpd - elpd)
    return 2 * (p_dic - lpd)
end

function kbnsum(s, c, x)
    t = s + x
    if abs(s) >= abs(x)
        c += (s - t) + x
    else
        c += (x - t) + s
    end
    return t, c
end
kbnsum(sc::NTuple{2,T}, x::T) where {T} = kbnsum(first(sc), last(sc), x)
kbnsum(sc::NTuple{2,T}) where {T} = sum(sc)
kbnsum(::Type{T}) where {T} = (zero(T), zero(T))

function waic(model::BayesianRegression, chains::AbstractChains{T}) where {T}
    p_waic = kbnsum(T)
    lppd = kbnsum(T)
    S = size(chains, 1) * size(chains, 2)
    logS = log(S)
    invS = inv(S)
    for (y, x) in zip(response(model), observations(model))
        ml = (T(-Inf), zero(T))     # Expected Likelihood over samples
        mll = kbnsum(T)             # Expected Log Likelihood over samples
        for I = CartesianIndices(axes(chains)[1:2])
            θ = chains[I.I..., :]
            ŷ = predict(model, θ, x)
            ℓ = deviance_logdensity(model, y, ŷ; θ)
            ml = LogExpFunctions._logsumexp_onepass_op(ℓ - logS, ml)
            mll = kbnsum(mll, ℓ)
        end

        # Accumulate into p_waic and lppd
        lppd_sample = LogExpFunctions._logsumexp_onepass_result(ml)
        p_waic = kbnsum(p_waic, lppd_sample)
        p_waic = kbnsum(p_waic, -kbnsum(mll) * invS)
        lppd = kbnsum(lppd, lppd_sample)
    end

    lppd = kbnsum(lppd)
    p_waic = kbnsum(p_waic)
    return 2 * (p_waic - lppd)
end

function score_model(model, chains::AbstractChains)
    ess, rhat = ess_rhat(chains)
    return (;
        aic=aic(model, chains),
        dic=dic(model, chains),
        waic=waic(model, chains),
        bic=bic(model, chains),
        ess_worse=minimum(values(ess)),
        rhat_worse=maximum(values(rhat)),
    )
end

function sample_model_score(model, chains; n=0.1)
    sc = subsample(chains, n)
    return score_model(model, sc)
end

function sample_model_score(outdir::String; kwargs...)
    data = jldopen(joinpath(outdir, "chains.jld2"), "r")
    model = data["model"]
    chains = data["chains"]
    return sample_model_score(model, chains; kwargs...)
end
