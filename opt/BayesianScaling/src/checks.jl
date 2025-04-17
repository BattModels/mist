"""
Compute correlations between the model's residuals and other possibly explanatory variables
"""
function residual_correlations(error::Vector, df)
    # Compute correlations between the residuals
    cols = names(df, eltype.(eachcol(df)) .<: Real)
    rescor = Dict{eltype(cols),Any}()
    for col in cols
        rescor[col] = StatsBase.corspearman(error, float.(df[!, col]))
    end
    return rescor
end

function StatsBase.aic(model::BayesianRegression, chains::AbstractChains)
    ℓ_mle = maximum(θ -> logpdf(:likelihood, model, θ), eachslice(chains; dims=(1, 2)))
    k = dimension(model)
    return 2 * (k - ℓ_mle)
end

function StatsBase.bic(model::BayesianRegression, chains::AbstractChains)
    ℓ_mle = maximum(θ -> logpdf(:likelihood, model, θ), eachslice(chains; dims=(1, 2)))
    k = dimension(model)
    n = length(observations(model))
    return 2 * (k * log(n) - ℓ_mle)
end

function StatsBase.rmsd(model::BayesianRegression, chains::AbstractChains)
    y = response(model)
    θ = maximum_posterior_estimate(model, chains)
    ŷ = predict(model, θ)
    return StatsBase.rmsd(y, ŷ)
end

function mape(model::BayesianRegression, chains::AbstractChains)
    y = response(model)
    θ = maximum_posterior_estimate(model, chains)
    ŷ = predict(model, θ)
    return mean(zip(ŷ, y)) do (ŷ, y)
        abs(ŷ - y) / y
    end
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
        mape=mape(model, chains),
        rmse=StatsBase.rmsd(model, chains),
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
