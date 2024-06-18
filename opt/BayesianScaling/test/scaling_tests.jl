@testitem "Hoffman Scaling" begin
    using BayesianScaling: hoffman_scaling, sample_posterior, compute_optimal_model, estimate_datasize
    using Turing: Prior, sample
    m = hoffman_scaling([1], [1], [1])
    θ = sample(m, Prior(), 1, progress=false)
    C = 1
    N_opt = only(sample_posterior(compute_optimal_model, θ, C))
    @test 0 <= N_opt

    # Check that we're at the optimum
    f = N -> only(sample_posterior(hoffman_scaling, θ, N, C / 6N))
    δ = 0.1
    @test f(N_opt) < f((1 - δ) * N_opt)
    @test f(N_opt) < f(δ * N_opt)

    # Check optimal model is consistent
    D_opt = estimate_datasize(N_opt, θ)
    @test only(D_opt) ≈ C / 6N_opt rtol = 1e-4
end

@testitem "Bayesian Scaling" begin
    using BayesianScaling: bayes_llm_scaling_model, hoffman_scaling, sample_posterior, compute_optimal_model, estimate_lr, estimate_datasize
    using Turing: Prior, sample

    m = bayes_llm_scaling_model([1], [1], [1], [1], [1], [1])
    θ = sample(m, Prior(), 1, progress=false)
    C = 1
    N_opt = only(sample_posterior(compute_optimal_model, θ, C))
    @test 0 <= N_opt

    # Check that we're at the Hoffman optimum
    f = N -> only(sample_posterior(hoffman_scaling, θ, N, C / 6N))
    δ = 0.1
    @test f(N_opt) < f((1 - δ) * N_opt)
    @test f(N_opt) < f(δ * N_opt)

    # Check optimal model is consistent
    D_opt = estimate_datasize(N_opt, θ)
    @test only(D_opt) ≈ C / 6N_opt rtol = 1e-4

    # Check LR
    lr = estimate_lr(N_opt, θ)
    @test only(lr) > 0
end
