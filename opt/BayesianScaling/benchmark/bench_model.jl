using BenchmarkTools
using BayesianScaling
using DataFrames: DataFrame
using Distributions: LogNormal, Exponential
using LogDensityProblems: logdensity, logdensity_and_gradient
using LogDensityProblemsAD: ADgradient

# AD Backends to Benchmark
using Zygote: Zygote
using ForwardDiff: ForwardDiff
using ReverseDiff: ReverseDiff
using Enzyme: Enzyme

const suite = BenchmarkGroup()

# Instantiate model with data
function init_hoffman(; nobs=1000, adtype=:Enzyme)
    df = DataFrame(;
        loss=rand(LogNormal(-2, 1), nobs),
        model_size=rand(Exponential(1e6), nobs),
        data_size=rand(Exponential(1e6), nobs),
        lr=rand(LogNormal(-4, 1), nobs),
    )
    model = BayesianScaling.HoffmanScaling(df)
    θ = BayesianScaling.sample_priors(model)
    model = BayesianScaling.init_logdensity_model(model, adtype)
    return model, θ
end

function init_progress_model(; adtype=:Enzyme, nsteps=100, nobs=300)
    df = DataFrame(;
        val_loss_trace=map(_ -> rand(LogNormal(-2, 1), nsteps), 1:nobs),
        rel_step_trace=map(_ -> collect(range(0, 1; length=nsteps + 2)[2:end-1]), 1:nobs),
        model_size=rand(Exponential(1e6), nobs),
        data_size=rand(Exponential(1e6), nobs),
        lr=rand(LogNormal(-4, 1), nobs),
        ff_ratio=rand([1, 2, 4], nobs),
        aspect_ratio=rand(16:50, nobs),
        eff_batch_size=128 .* rand(32:64, nobs) .* rand([16, 32, 64], nobs),
    )
    model = BayesianScaling.TrainingProgress(df)
    θ = BayesianScaling.sample_priors(model)
    model = BayesianScaling.init_logdensity_model(model, adtype)
    return model, θ
end

models = [
    "training_progress" => init_progress_model,
    "hoffman_scaling" => init_hoffman,
]
for (name, instantiate) in models
    s = suite[name] = BenchmarkGroup()
    s["logdensity"] = @benchmarkable logdensity(args...) setup = (args = $instantiate())
    for ad in [:ForwardDiff, :ReverseDiff, :Zygote, :Enzyme]
        s["$ad"] = @benchmarkable logdensity_and_gradient(args...) setup = (args = $instantiate(; adtype=Symbol($ad)))
    end
end
