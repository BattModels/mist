abstract type AcquisitionFunction end

""" Curry an acquisition function `aq` with a Bayesian-objective function `f` """
(aq::AcquisitionFunction)(f) = (θ, x...) -> aq(f, θ, x...)

struct UpperConfidenceBound{T<:Real} <: AcquisitionFunction
    λ::T
end
function (aq::UpperConfidenceBound)(model, θ, x...)
    y = sample_posterior(model.f, θ, x...)
    μ, σ = mean_and_std(y)
    return μ + aq.λ * σ
end

struct LowerConfidenceBound{T<:Real} <: AcquisitionFunction
    λ::T
end
function (aq::LowerConfidenceBound)(model, θ, x...)
    y = sample_posterior(model.f, θ, x...)
    μ, σ = mean_and_std(y)
    return -(μ - aq.λ * σ) / log(*(6, x...))
end

struct BayesianAssurance{T<:Real}
    δ::T
    w::T
    K::T
end
function linear_cost_effectiveness(model, K, θ, x...)
    utility = K * mean(sample_posterior(model.f, θ, x...))
    cost = sum(*(6, x...))
    return utility - cost
end

struct ShannonInformation <: AcquisitionFunction
    N::Int
    M::Int
end
function (aq::ShannonInformation)(model, chains, x...)
    info_gain = 0
    (; N, M) = aq
    y_varname = first(keys(model.args))
    # Drawn N paired samples of y, θ from the Prior
    model_gen = model.f(missing, x...)
    ℓ = Turing.LogDensityFunction(model_gen)
    ns = size(chains, 1)
    N = round(Int, ns * N / (N * M))
    M = fld(ns, N)


    info_gain = 0
    for i in sample(1:ns, N, replace=false)
        # Sample y_i, and evaluate -I(y| θ, x)
        chain = chains[i, :, 1]
        y_i = first(rand(Turing.condition(model_gen, chain)))
        θ = vec(Array(chain))
        expr_info = logdensity(ℓ, vcat(θ, y_i))

        # Sample M values of θ, and estimate I(y|x)
        data_info = map(sample(1:ns, M, replace=false)) do j
            θ = vec(Array(chains[j, :, 1]))
            exp(logdensity(ℓ, vcat(θ, y_i)))
        end
        mu = mean(data_info)
        if mu > 0
            info_gain += expr_info - log(mu)
        else
            N -= 1
        end
    end

    return info_gain / N
end
