@testitem "sampleposterior" begin
    using BayesianScaling: sample_posterior
    using Distributions: Normal
    using Turing: Chains
    using MLUtils: stack

    chains = Chains(rand(Normal(), 2, 3, 4), [:a, :b, :c])
    @test size(chains) == (2, 3, 4)
    f = (scale; a, b, c) -> @. scale * hypot(a, b) + c
    s = sample_posterior(f, chains, 2)
    @test size(s) == (2, 4,) # Flatten Chains
    x = [1, 2, 4]
    s = sample_posterior(f, chains, x)
    @test size(s) == (3, 2, 4)
    @test s == stack(map(x -> sample_posterior(f, chains, x), x); dims=1)
end
