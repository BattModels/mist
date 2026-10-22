using Test
using PythonCall
using Mixtures

@testset "Mixture Python result conversion" begin
    row = pyimport("json").loads(raw"""{
        "compounds": ["O", "CCO"], "composition": [0.25, 0.75], "temperature": 298.15,
        "y": [1.0, 2.0], "y_excess": [0.1, 0.2], "y_linear": [0.9, 1.8],
        "target": [1.1, 0.0], "target_mask": [true, false],
        "target_excess": [0.0, 0.3], "target_excess_mask": [false, true],
        "dy_dT": [0.01, 0.02], "dy_excess_dT": [0.001, 0.002],
        "dy_dx": [[1.0, 2.0], [3.0, 4.0]],
        "dy_excess_dx": [[0.1, 0.2], [0.3, 0.4]]
    }""")
    # The Python evaluator returns torch tensors for gradient fields.
    row["dy_dx"] = pyimport("torch").tensor(row["dy_dx"])
    out = Mixtures.excess_row(row, ["density", "molar_volume"]; references=true, gradients=true)
    @test out["compounds"] == ["O", "CCO"]
    @test out["composition"] == [0.25, 0.75]
    @test out["density"] == 1.0
    @test out["molar_volume"] == 2.0
    @test out["density_ref"] == 1.1
    @test ismissing(out["molar_volume_ref"])
    @test ismissing(out["density_excess_ref"])
    @test out["molar_volume_excess_ref"] == 0.3
    @test out["density_dx"] == [1.0, 2.0]
    @test out["molar_volume_dx"] == [3.0, 4.0]
    @test out["density_dT"] == 0.01
    @test out["molar_volume_excess_dx"] == [0.3, 0.4]
    plain = Mixtures.excess_row(row, ["density", "molar_volume"])
    @test !haskey(plain, "density_ref")
    @test !haskey(plain, "density_dx")
end

@testset "Ionic Python result conversion" begin
    row = pyimport("json").loads(raw"""{
        "components": ["O", "CCO", "CC", "[Cl-]", "[Li+]"],
        "composition": [0.3, 0.3, 0.3, 0.1], "temperature": 298.15,
        "conductivity": 0.5, "ln_A": 2.0, "Ea": 100.0, "Tg": 200.0,
        "alpha": 0.1, "beta": 0.2, "lmbda": 0.3
    }""")
    out = Mixtures.conductivity_row(row)
    @test out["components"] == ["O", "CCO", "CC", "[Cl-]", "[Li+]"]
    @test out["composition"] == [0.3, 0.3, 0.3, 0.1]
    @test out["conductivity"] == 0.5
    @test out["Ea"] == 100.0
    @test out["Tg"] == 200.0
    @test out["lmbda"] == 0.3
end
