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

    # Draw samples for θ and y from the chains
    in_i = sample(1:ns, N, replace=false)
    y_mu = sample_posterior(model.f, chains, x...)
    y = rand(MvLogNormal(y_mu, diagm(vec(Array(chains[:σ²])))))


    θ = Array(chains)
    info_gain = Normal(0, 0)
    for i in in_i
        expr_info = logdensity(ℓ, vcat(θ[i, :], y[i]))
        in_j = sample(1:ns, M, replace=false)
        data_info = map(in_j) do j
            exp(logdensity(ℓ, vcat(θ[j, :], y[j])))
        end
        mu = mean(data_info)
        data_info = Normal(log(mu), abs(std(data_info) / mu))
        info_gain = convolve(info_gain, expr_info + (-data_info))
    end

    return info_gain / N



end

#     for i in 1:N
#         θ_i = @view θ[i, :]
#         y_i = sample(
#         y_i = predict(model_gen, θ[i, :])["loss"]
#         expr_info = logdensity(ℓ, selectdim(θ, 1, i))
#         data_info = 0.0
#         for j in 1:M
#             s_work[1:np] .= θ[i, :] # Use sampled value of θ
#             s_work[end] = θ[i, end] # Use sampled value of y_i
#             data_info += exp(logdensity(ℓ, s_work))
#         end
#         data_info /= M
#         if data_info > 0
#             info_gain += (expr_info - log(data_info))
#         else
#             N -= 1
#         end
#     end
#     return info_gain / N
# end
