using Test
using CSV
using DataFrames
using Makie
using MISTStyle
using Mixtures

module IonicTablesUnderTest
include("../ionic_conductivity.jl")
end

module IonicPanelsUnderTest
include("../panel.jl")
end

struct IonicTableFixture end

# Exercise the real grid and table code without loading or evaluating a model.
function Mixtures.evaluate_at_composition(::IonicTableFixture, mixture::Dict, composition::Vector{Float64})
    @assert sum(composition) ≈ 1.0
    @assert composition[2:3] == [0.0, 0.0]
    @assert mixture["salt"][2] == "[Li+]"
    return Dict("conductivity" => mixture["temperature"] / 100 + last(composition), "Tg" => 200.0)
end

@testset "Generated ionic figure artifacts" begin
    curves, inference = IonicTablesUnderTest.conductivity_tables(IonicTableFixture())
    @test nrow(curves) == 500
    @test unique(curves.Solvent) == ["PC"]
    @test Set(curves.salt_name) == Set(["LiPF6", "LiTFSI"])
    for group in groupby(curves, [:salt_name, :temperature])
        @test nrow(group) == 50
        @test issorted(group.composition)
        @test first(group.composition) == 0.02
        @test last(group.composition) == 0.20
    end
    @test unique(curves.temperature) == collect(260:20:340)
    @test curves.predictions ≈ exp.(curves.temperature ./ 100 .+ curves.composition)

    @test nrow(inference) == 102
    @test Set(inference.Solvent) == Set(["FEC", "PC", "EMC", "EC", "DEC", "DMC"])
    @test unique(inference.salt_name) == ["LiPF6"]
    @test unique(inference.molarity) == [1.0]
    for group in groupby(inference, :Solvent)
        @test group.temperature == collect(260:5:340)
        @test unique(group.composition) == [Mixtures.molarity_to_mole_fraction(first(group.Solvent); molarity=1.0)]
    end
    @test inference[!, "Tg/T"] ≈ 200.0 ./ inference.temperature
    @test inference.conductivity ≈ inference.temperature ./ 100 .+ inference.composition

    mktempdir() do output
        for (filename, table) in [("ionic_conductivity_curves.csv", curves),
                                  ("ionic_conductivity_inference.csv", inference)]
            path = joinpath(output, filename)
            CSV.write(path, table)
            @test DataFrame(CSV.File(path)) == table
        end
        with_theme(MISTStyle.theme()) do
            figure = Figure()
            IonicPanelsUnderTest.plot_ionic_conductivity!(figure[1, 1];
                data_path=joinpath(output, "ionic_conductivity_curves.csv"), generated=true)
            axis = content(figure[1, 1][1, 1])
            @test axis.limits[][2][2] > maximum(curves.predictions)
            curves_drawn = filter(plot -> plot isa Lines, axis.scene.plots)
            @test length(curves_drawn) == 10
            @test all(length(plot[1][]) == 50 for plot in curves_drawn)
            angell = IonicTablesUnderTest.plot_angell_solvents(;
                data_path=joinpath(output, "ionic_conductivity_inference.csv"), generated=true)
            @test content(angell[1, 1]).xlabel[] == L"T_0/T"
        end
    end
end
