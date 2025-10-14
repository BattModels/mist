using Test
using Mixtures

@testset "weighted_ternary_similarity" begin

    @testset "Binary mixture - same molecule" begin
        solvents = ["O=C1OCCO1", "O=C1OCCO1"]
        compositions = [0.3, 0.4]

        result = weighted_ternary_similarity(solvents, compositions)

        expected = 1.0
        @test result ≈ expected
    end

    @testset "Ternary mixture - same molecule" begin
        solvents = ["O=C1OCCO1", "O=C1OCCO1", "O=C1OCCO1"]
        compositions = [0.1, 0.1, 0.1]
        result = weighted_ternary_similarity(solvents, compositions)

        sim12 = get_similarity(solvents[1], solvents[2])
        sim13 = get_similarity(solvents[1], solvents[3])
        sim23 = get_similarity(solvents[2], solvents[3])

        expected = 1.0

        @test sim12 ≈ expected
        @test sim13 ≈ expected
        @test sim23 ≈ expected
        @test result ≈ expected
    end

    @testset "Ternary mixture - unequal compositions" begin
        solvents = ["O=C1OCCO1", "O=C1OCC(F)O1", "CCOC(=O)OCC"]
        compositions = [0.2, 0.3, 0.1]

        result = weighted_ternary_similarity(solvents, compositions)

        sim12 = get_similarity(solvents[1], solvents[2])
        sim13 = get_similarity(solvents[1], solvents[3])
        sim23 = get_similarity(solvents[2], solvents[3])

        expected = 0.5*((0.5/0.6) * sim12 + (0.3/0.6) * sim13 + (0.4/0.6) * sim23)
        @test result ≈ expected
    end


    @testset "Zero composition handling" begin
        solvents = ["O=C1OCCO1", "COC(=O)OC", "O=C1OCC(F)O1"]
        compositions = [0.3, 0.3, 0.0]

        result = weighted_ternary_similarity(solvents, compositions)

        sim12 = get_similarity(solvents[1], solvents[2])
        sim13 = get_similarity(solvents[1], solvents[3])
        sim23 = get_similarity(solvents[2], solvents[3])

        expected = (sim12 + 0.5 * sim13 + 0.5 * sim23)/2.0
        @test result ≈ expected
    end
end
