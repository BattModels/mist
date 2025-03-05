@testitem "compenstated sum" begin
    using BayesianScaling: kbnsum
    s, c = kbnsum((1e8, 0.0), 1e-6)
    @test s > 1e8
    @test c > 0
    @test s ≈ 1e8 + 1e-6 atol = c
    x = rand(1000)
    s = kbnsum(foldl(kbnsum, x; init=kbnsum(eltype(x))))
    s_ref = sum(x)
    @test s ≈ s_ref

    @test @timed(kbnsum((0.3, 0.0), 0.3)).bytes == 0
end

@testitem "BayesianRegression" begin
    using StatsBase
    using LogDensityProblems
    using Enzyme
    using Distributions: Normal, Exponential, logpdf
    using BayesianScaling: BayesianScaling, BayesianRegression
    using TransformVariables: transform

    # Construct Model
    formula = (x, θ) -> x * θ.a + θ.b
    priors = (;
        a = Normal(0, 2),
        b = Normal(0, 2),
        sigma = Exponential(1.0),
    )
    observations = 10*rand(100)
    response = @. 3.2 * observations + 1.2
    response .+= 0.1 * randn(length(response))
    model = BayesianRegression(formula, priors, response, observations, Normal())

    @testset "interface" begin
        θ = (; a=3.0, b=2.0, sigma=0.2)
        @test BayesianScaling.predict(model, θ, 1.2) == (1.2 * 3.0 + 2.0)
        @test BayesianScaling.deviance_logdensity(model, 2.1, 1.0; θ) isa Real
    end

    @testset "credible_prior" begin
        lb, ub = BayesianScaling.credible_prior(model)
        @test length(lb) == length(ub) == 3
        @test all(lb .< ub)
    end

    @testset "logdensity" begin
        θ = (; a=2.1, b=0.3, sigma=0.3)
        ℓ_prior = logpdf(:prior, model, θ)
        ℓ_data = logpdf(:likelihood, model, θ)
        ℓ = logpdf(:joint, model, θ)
        @test ℓ_prior isa Real
        @test ℓ_data isa Real
        @test ℓ isa Real
        @test ℓ == (ℓ_prior + ℓ_data)

        @test logpdf(:likelihood, model, (; a=3.2, b=1.2, sigma=0.1)) >= 10
    end

    @testset "LogDensityProblems" begin
        ℓ = BayesianScaling.init_logdensity_model(model)
        @test LogDensityProblems.dimension(ℓ) == 3
        funs = [
            LogDensityProblems.logdensity,
            LogDensityProblems.logdensity_and_gradient,
        ]
        @testset "$f" for f in funs
            @test LogDensityProblems.stresstest(f, ℓ) |> isempty
        end
    end

    @testset "map estimate" begin
        θ_map, sol = BayesianScaling.maximum_posterior_estimate(model)
        @test θ_map isa NamedTuple
        @test θ_map.a ≈ 3.2 rtol=1e-1
        @test θ_map.b ≈ 1.2 rtol=1e-1
        @test θ_map.sigma ≈ 0.1 rtol=1e-1
    end

    @testset "Bayesian estimate" begin
        chains = BayesianScaling.sample_chains(model; nchains=4, draws=100)
        @test chains isa BayesianScaling.AbstractChains
        @test size(chains) == (100, 4, 3)
        θ_exp = mean(eachslice(chains; dims=(1,2)))
        @test keys(θ_exp) == (:a, :b, :sigma)
        @test θ_exp ≈ [3.2, 1.2, 0.1] rtol=1e-1
        θ_map = BayesianScaling.maximum_posterior_estimate(model, chains)
        @test keys(θ_exp) == (:a, :b, :sigma)
        @test θ_exp ≈ [3.2, 1.2, 0.1] rtol=1e-1
    end
end
