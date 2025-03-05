@testitem "Hoffman Scaling" begin
    using DataFrames
    using StatsBase: mean
    using Enzyme: Enzyme
    using LogDensityProblems: LogDensityProblems
    using BayesianScaling: BayesianScaling, BayesianRegression, HoffmanScaling, hoffman_scaling
    using Random: seed!

    # Generate some data
    seed!(1)
    n = 1000
    df = DataFrame(
        model_size=rand(10:10_000, n),
        data_size=rand(100:100_000, n),
    )
    df.loss = @. hoffman_scaling(df.model_size, df.data_size; A=400, B=410, α=0.34, β=0.37, E=1.7)
    df.loss .+= 0.01 .* rand(n)

    function check_solution(θ; rtol=0.05)
        @test isapprox(θ.A, 400; rtol)
        @test isapprox(θ.B, 410; rtol)
        @test isapprox(θ.α, 0.34; rtol)
        @test isapprox(θ.β, 0.37; rtol)
        @test isapprox(θ.E, 1.7; rtol)
        return nothing
    end

    model = BayesianScaling.init_model(HoffmanScaling(), df)

    @testset "huber fit" begin
        θ_huber, sol = BayesianScaling.fit_model_huber(model)
        @test Symbol(sol.retcode) == :Success
        check_solution(θ_huber)
    end

    @testset "map estiamte" begin
        θ_map, sol = BayesianScaling.maximum_posterior_estimate(model; maxiters=10)
        @test Symbol(sol.retcode) == :MaxIters
    end

    @testset "performance" begin
        L = BayesianScaling.init_logdensity_model(model)
        @test @inferred(Float64, BayesianScaling.logdensity(L, rand(6))) isa Float64
        d = @timed LogDensityProblems.logdensity(L, rand(6))
        @test d.time <= 1e-3
        @test d.recompile_time == 0
        @test d.compile_time == 0
        @test d.bytes <= 256

        LogDensityProblems.logdensity_and_gradient(L, rand(6))
        d = @timed LogDensityProblems.logdensity_and_gradient(L, rand(6))
        @test d.time <= 1e-2
        @test d.compile_time == 0
        @test d.recompile_time == 0
    end
end
