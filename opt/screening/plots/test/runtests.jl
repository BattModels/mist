using Test
using ScreeningPlots
using PythonCall: pyimport, PyList, pybuiltins
using SQLite
using DataFrames

module Supplementary
include("../supplementary.jl")
end

@testset "Model channel schemas" begin
    namespace = pyimport("types").SimpleNamespace
    strings = namespace(channels=PyList(["homo", "gap"]))
    schemas = namespace(channels=PyList([
        pybuiltins.dict(name="homo", units="eV"),
        pybuiltins.dict(name="gap", units="eV"),
    ]))
    @test ScreeningPlots.channel_names(strings) == ["homo", "gap"]
    @test ScreeningPlots.channel_names(schemas) == ["homo", "gap"]
end

@testset "Parent-adjusted halogen coefficients" begin
    rows = DataFrame(parent_id=String[], halogen_type=String[], value=Float64[])
    for parent in 1:12, (index, halogen) in enumerate(["F", "Cl", "Br", "I", "F-Cl"])
        push!(rows, (string(parent), halogen, 2parent + 0.6index + 0.05sin(parent * index)))
    end
    for property in ["homo", "lumo", "gap", "bp", "mp", "fp", "Cv"]
        rows[!, property] = rows.value
    end
    fitted = Supplementary.halogen_coefficients(rows)
    @test nrow(fitted) == 7 * 4
    @test !("F" in fitted.term)
    chlorine = subset(fitted, :term => ByRow(==("Cl")))
    @test all(isapprox.(chlorine.estimate, 0.6; atol=0.05))
    @test all(chlorine.lower .< chlorine.estimate .< chlorine.upper)
    @test all(chlorine.p_value .< 0.05)
    @test_throws ErrorException Supplementary.halogen_coefficients(rows; baseline="absent")
    @test_throws ErrorException Supplementary.halogen_coefficients(select(rows, Not(:parent_id)))
    mktempdir() do folder
        figure = Supplementary.coefficient_figure(fitted, ["bp", "mp", "fp", "Cv"], (2, 2))
        Supplementary.MISTStyle.savefig("fixture", figure; fig_dir=folder)
        @test filesize(joinpath(folder, "fixture.pdf")) > 1000
        @test filesize(joinpath(folder, "fixture.png")) > 1000
    end
end

@testset "Screening database input" begin
    mktempdir() do folder
        path = joinpath(folder, "screen.sqlite")
        db = SQLite.DB(path)
        SQLite.DBInterface.execute(db, "CREATE TABLE molecules (inchi TEXT, smiles TEXT, props TEXT)")
        close(db)
        @test_throws ArgumentError ScreeningPlots.load_generated_molecules(path)
        db = SQLite.DB(path)
        SQLite.DBInterface.execute(db, "INSERT INTO molecules VALUES (?, ?, ?)",
            ("example-key", "CCO", "{\"homo\": -6.0, \"gap\": 7.0}"))
        close(db)
        rows = ScreeningPlots.load_generated_molecules(path)
        @test rows.inchi_key == ["example-key"]
        @test rows.homo == [-6.0]
        @test rows.gap == [7.0]
    end
end
