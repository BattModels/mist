@testitem "penalties" begin
    using BayesianScaling: harmonic_penalty, geometric_penalty, symmetric_polynomial

    @testset "symmetric_polynomial" begin
        @test 0 ≈ symmetric_polynomial(0, 1)
        @test 1 ≈ symmetric_polynomial(1, 1)
        @test 4 ≈ symmetric_polynomial(1, 4)
        @test symmetric_polynomial(1, 4) ≈ symmetric_polynomial(-1, 4)
        ref = (x, coef) -> sum(i_c -> i_c[2] * x^(2 * i_c[1]), enumerate(coef))
        coefs = (2, 3)
        @test ref(0, coefs) isa Real
        @test ref(0, coefs) ≈ symmetric_polynomial(0, coefs...)
        @test ref(1, coefs) ≈ symmetric_polynomial(1, coefs...)
        @test ref(-1, coefs) ≈ symmetric_polynomial(-1, coefs...)
    end

    @test geometric_penalty(1, 0, 1, 3) >= 0
    @test harmonic_penalty(1, 0, 1, 3) >= 0
    @test geometric_penalty(1, 0, 1, 3) >= 0
end
