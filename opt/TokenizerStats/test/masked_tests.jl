@testitem "Construction" begin
    using TokenizerStats: MaskedCode

    mc = MaskedCode(rand(1:32, 10), rand(Bool, 10), -100)
    @test length(mc) == 10
    @test eltype(mc) == Int
    @test size(mc) == (10,)

    mcv = view(mc, 1:5)
    @test length(mcv) == 5
    @test eltype(mcv) == Int

    # Check MaskedCode assertions hold
    @test_throws AssertionError MaskedCode(rand(1:32, 10), rand(Bool, 9))
    @test_throws AssertionError MaskedCode([1, 2, 3, 4], falses(4), 3)
    @test_throws InexactError MaskedCode(rand(UInt32, 10), rand(Bool, 10), -100)
end

@testitem "indexing" begin
    using TokenizerStats: MaskedCode
    mc = MaskedCode(rand(1:32, 10), falses(10), -100)
    @test mc[1] == mc.code[1]
    @test mc[1:5] == mc.code[1:5]
end

@testitem "UInt32" begin
    using TokenizerStats: MaskedCode
    mc = MaskedCode(rand(UInt32, 10), falses(10))
    @test mc[1:5] == mc.code[1:5]
end
