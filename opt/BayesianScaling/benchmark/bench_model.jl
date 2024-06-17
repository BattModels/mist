using BenchmarkTools
using BayesianScaling
using Distributions: LogNormal, Exponential
using Turing: LogDensityFunction, Prior, sample
using LogDensityProblems: logdensity, logdensity_and_gradient
using LogDensityProblemsAD: ADgradient

# AD Backends to Benchmark
using Zygote: Zygote
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff

const suite = BenchmarkGroup()

# Instantiate model with data
function instantiate_model()
    nobs = 1000
    loss = rand(LogNormal(-2, 1), nobs)
    model_size = rand(Exponential(1e6), nobs)
    data_size = rand(Exponential(1e6), nobs)
    lr = rand(LogNormal(-4, 1), nobs)
    ff_ratio = rand([1, 2, 4], nobs)
    aspect_ratio = rand(16:50, nobs)
    model = BayesianScaling.bayes_llm_scaling_model(loss, model_size, data_size, lr, ff_ratio, aspect_ratio)
    θ = vec(Array(sample(model, Prior(), 1, progress=false)))
    return model, θ
end

function for_density()
    model, θ = instantiate_model()
    ℓ = LogDensityFunction(model)
    return ℓ, θ
end

function for_gradient(ad)
    model, θ = instantiate_model()
    ℓ = ADgradient(Symbol(ad), LogDensityFunction(model))
    return ℓ, θ
end


suite["model"]["logdensity"] = @benchmarkable logdensity(args...) setup = (args = for_density())
for ad in [:ForwardDiff, :Zygote, :ReverseDiff]
    suite["model"]["$ad"] = @benchmarkable logdensity_and_gradient(args...) setup = (args = for_gradient($(ad)))
end
