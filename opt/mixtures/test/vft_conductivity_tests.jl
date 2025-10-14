using Test
using DataFrames
using Mixtures: calculate_excess!

@testset "calculate_excess!" begin

    @testset "Binary mixture" begin
        # 3 pure solvents + 1 binary mixture
        df = DataFrame(
            composition = [
                [0.9, 0.0, 0.0, 0.1],
                [0.0, 0.9, 0.0, 0.1],
                [0.0, 0.0, 0.9, 0.1],
                [0.4, 0.4, 0.0, 0.1],
            ],
            Ea = [10.0, 20.0, 10.0, 15.0],
            Tg = [200.0, 300.0, 210.0, 250.0],
        )

        result = calculate_excess!(df)

        @test hasproperty(result, :ideal_mixing_Ea)
        @test hasproperty(result, :excess_Ea)
        @test hasproperty(result, :relative_excess_Ea)
        @test hasproperty(result, :ideal_mixing_Tg)
        @test hasproperty(result, :excess_Tg)
        @test hasproperty(result, :relative_excess_Tg)

        expected_Ea = 0.5 * 10.0 + 0.5 * 20.0
        @test result[4, :ideal_mixing_Ea] ≈ expected_Ea

        expected_Tg = 0.5 * 200.0 + 0.5 * 300.0
        @test result[4, :ideal_mixing_Tg] ≈ expected_Tg

        # Check excess calculation
        @test result[4, :excess_Ea] ≈ 15.0 - expected_Ea
        @test result[4, :excess_Tg] ≈ 250.0 - expected_Tg

        # Check degenerate cases
        @test all(abs.(result[1:4, :excess_Ea]) .< 1e-3)
        @test all(abs.(result[1:4, :excess_Tg]) .< 1e-3)
    end

    @testset "Ternary mixture ideal mixing" begin
        df = DataFrame(
            composition = [
                [0.9, 0.0, 0.0, 0.1],
                [0.0, 0.9, 0.0, 0.1],
                [0.0, 0.0, 0.9, 0.1],
                [0.3, 0.3, 0.3, 0.1],
            ],
            Ea = [10.0, 20.0, 30.0, 20.0],
            Tg = [100.0, 200.0, 300.0, 200.0],
        )

        result = calculate_excess!(df)

        expected_Ea = (10.0 + 20.0 + 30.0) / 3.0
        @test result[4, :ideal_mixing_Ea] ≈ expected_Ea

        expected_Tg = (100.0 + 200.0 + 300.0) / 3.0
        @test result[4, :ideal_mixing_Tg] ≈ expected_Tg

        # Check excess
        @test abs(result[4, :excess_Ea] - (20.0 - expected_Ea)) < 1e-10
        @test abs(result[4, :excess_Tg] - (200.0 - expected_Tg)) < 1e-10
    end

    @testset "Ternary mixture positive excess" begin
        df = DataFrame(
            composition = [
                [0.9, 0.0, 0.0, 0.1],
                [0.0, 0.9, 0.0, 0.1],
                [0.0, 0.0, 0.9, 0.1],
                [0.4, 0.4, 0.1, 0.1],
            ],
            Ea = [10.0, 20.0, 15.0, 20.0],
            Tg = [200.0, 300.0, 250.0, 275.0],
        )

        result = calculate_excess!(df)

        # Ideal mixing
        expect_Ea = 10.0*(4/9) + 20.0*(4/9) + 15.0*(1/9)
        @test result[4, :ideal_mixing_Ea] ≈ expect_Ea
        expect_Tg = 200.0*(4/9) + 300.0*(4/9) + 250.0*(1/9)
        @test result[4, :ideal_mixing_Tg] ≈ expect_Tg

        # Positive excess
        @test result[4, :excess_Ea] ≈ 20.0 - expect_Ea
        @test result[4, :excess_Tg] ≈ 275.0 - expect_Tg

    end

    @testset "Ternary mixture negative excess" begin
        df = DataFrame(
            composition = [
                [0.9, 0.0, 0.0, 0.1],
                [0.0, 0.9, 0.0, 0.1],
                [0.0, 0.0, 0.9, 0.1],
                [0.4, 0.4, 0.1, 0.1],
            ],
            Ea = [10.0, 20.0, 15.0, 12.0],
            Tg = [200.0, 300.0, 250.0, 125.0],
        )

        result = calculate_excess!(df)

        expect_Ea = 10.0*(4/9) + 20.0*(4/9) + 15.0*(1/9)
        @test result[4, :excess_Ea] ≈ 12.0 - expect_Ea
        expect_Tg = 200.0*(4/9) + 300.0*(4/9) + 250.0*(1/9)
        @test result[4, :excess_Tg] ≈ 125.0 - expect_Tg

        @test result[4, :relative_excess_Ea] ≈ abs(12.0 - expect_Ea) / 12.0
        @test result[4, :relative_excess_Tg] ≈ abs(125.0 - expect_Tg) / 125.0
    end

end
