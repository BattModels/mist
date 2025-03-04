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
