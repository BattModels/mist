using BenchmarkTools
using Turing: LogDensityFunction
using LogDensityProblems: logdensity, logdensity_and_gradient
using LogDensityProblemsAD: ADgradient
using Zygote: Zygote

suite = BenchmarkGroup()

model = fit_models(df)
f = LogDensityFunction(model)

suite["model"]["logdensity"] = @benchmarkable logdensity($f, rand(13))

for ad in [:ForwardDiff, :Zygote, :ReverseDiff]
    f_ad = ADgradient(ad, f)
    suite["model"]["$ad"] = @benchmarkable logdensity_and_gradient($f_ad, rand(13))
end
