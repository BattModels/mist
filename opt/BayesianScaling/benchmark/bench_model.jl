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
    return model
end

function init_progress_model()
    nobs = 300
    nsteps = 100
    loss_trace = map(_ -> rand(LogNormal(-2, 1), nsteps), 1:nobs)
    rel_step = map(_ -> collect(range(0, 1; length=nsteps + 2)[2:end-1]), 1:nobs)
    model_size = rand(Exponential(1e6), nobs)
    data_size = rand(Exponential(1e6), nobs)
    lr = rand(LogNormal(-4, 1), nobs)
    ff_ratio = rand([1, 2, 4], nobs)
    aspect_ratio = rand(16:50, nobs)
    eff_batch_size = 128 .* rand(32:64, nobs) .* rand([16, 32, 64], nobs)
    model = BayesianScaling.training_progress(
        loss_trace,
        rel_step,
        model_size,
        data_size,
        lr,
        ff_ratio,
        aspect_ratio,
        eff_batch_size,
    )
    return model
end

get_theta(model) = vec(Array(sample(model, Prior(), 1, progress=false)))

function for_density(model)
    θ = get_theta(model)
    ℓ = LogDensityFunction(model)
    return ℓ, θ
end

function for_gradient(model, ad)
    θ = get_theta(model)
    ℓ = ADgradient(Symbol(ad), LogDensityFunction(model))
    return ℓ, θ
end


models = [
    "bayes_llm_scaling_model" => instantiate_model,
    "training_progress" => init_progress_model,
]
for (name, instantiate) in models
    s = suite[name] = BenchmarkGroup()
    s["logdensity"] = @benchmarkable logdensity(args...) setup = (args = for_density($instantiate()))
    for ad in [:ForwardDiff, :ReverseDiff]
        s["$ad"] = @benchmarkable logdensity_and_gradient(args...) setup = (args = for_gradient($instantiate(), $(ad)))
    end
end
