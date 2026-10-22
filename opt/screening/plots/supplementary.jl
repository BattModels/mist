#!/usr/bin/env -S julia --startup-file=no --project=@script
using ArgParse
using CSV
using DataFrames
using CategoricalArrays
using GLM
using CairoMakie
using MISTStyle
using PythonCall: pyimport, pyconvert

function require_columns(df, columns, source)
    absent = setdiff(columns, names(df))
    isempty(absent) || error("Missing columns in $source: $(join(absent, ", "))")
    isempty(df) && error("No observations in $source")
end

function halogen_coefficients(df; baseline="F")
    properties = ["homo", "lumo", "gap", "bp", "mp", "fp", "Cv"]
    require_columns(df, ["parent_id", "halogen_type", properties...], "mixed-halogen descriptors")
    results = DataFrame(property=String[], term=String[], estimate=Float64[],
        lower=Float64[], upper=Float64[], p_value=Float64[])
    for property in properties
        rows = dropmissing(select(df, :parent_id, :halogen_type, property => :value))
        all(isfinite, rows.value) || error("Nonfinite $property observations")
        rows.parent_id = categorical(string.(rows.parent_id))
        rows.halogen_type = categorical(string.(rows.halogen_type))
        baseline in levels(rows.halogen_type) || error("Missing fully fluorinated baseline '$baseline'")
        levels!(rows.halogen_type, [baseline; filter(!=(baseline), levels(rows.halogen_type))])
        # Paper SI: categorical halogen type with parent-scaffold fixed effects.
        fitted = lm(@formula(value ~ halogen_type + parent_id), rows)
        intervals = confint(fitted)
        probabilities = coeftable(fitted).cols[4]
        for (index, name) in enumerate(coefnames(fitted))
            startswith(name, "halogen_type: ") || continue
            push!(results, (property, replace(name, "halogen_type: " => ""),
                coef(fitted)[index], intervals[index, 1], intervals[index, 2], probabilities[index]))
        end
    end
    return results
end

function coefficient_figure(coefficients, properties, shape)
    figure = Figure(; size=(6.5inch, shape[1] * 2.5inch))
    for (index, property) in enumerate(properties)
        significant = subset(coefficients, :property => ByRow(==(property)), :p_value => ByRow(<(0.05)))
        row, column = divrem(index - 1, shape[2])
        axis = Axis(figure[row + 1, column + 1];
            title=property, xlabel="Coefficient relative to fully fluorinated parent",
            yticks=(1:nrow(significant), significant.term))
        vlines!(axis, [0.0]; color=:gray, linestyle=:dash)
        if !isempty(significant)
            errorbars!(axis, significant.estimate, 1:nrow(significant),
                significant.estimate - significant.lower, significant.upper - significant.estimate;
                direction=:x, color=:black)
            scatter!(axis, significant.estimate, 1:nrow(significant); color=:black)
        end
    end
    return figure
end

function calibration_figure(df)
    require_columns(df, ["solvent", "beta", "ETN", "Li_vs_SHE", "O2_vs_SHE"], "lithium-air calibration")
    equations = pyimport("opt.screening.lithium_air_equations")
    figure = Figure(; size=(6.5inch, 2.7inch))
    for (index, descriptor, potential, model) in [
        (1, :beta, :Li_vs_SHE, equations.f_SHE_Li),
        (2, :ETN, :O2_vs_SHE, equations.f_O2_O2minus),
    ]
        rows = dropmissing(select(df, descriptor, potential))
        nrow(rows) >= 3 || error("At least three original calibration observations required for $potential")
        all(isfinite, rows[!, descriptor]) && all(isfinite, rows[!, potential]) || error("Nonfinite calibration observations")
        axis = Axis(figure[1, index]; xlabel=string(descriptor), ylabel=string(potential) * " (V)")
        scatter!(axis, rows[!, descriptor], rows[!, potential]; color=:black)
        grid = range(extrema(rows[!, descriptor])...; length=200)
        lines!(axis, grid, [pyconvert(Float64, model(value)) for value in grid])
    end
    return figure
end

function main(args)
    settings = ArgParseSettings(description="Render screening SI from original descriptor/calibration observations.")
    @add_arg_table! settings begin
        "--analysis"
            required = true
            range_tester = x -> x in ["halogens", "calibration"]
        "--input"
            required = true
        "--output-dir"
            required = true
        "--baseline"
            default = "F"
    end
    options = parse_args(args, settings)
    source = options["input"]
    isfile(source) || error("Missing original $(options["analysis"]) observations: $source. See opt/screening/README.md.")
    df = DataFrame(CSV.File(source))
    output = options["output-dir"]
    mkpath(output)
    with_theme(MISTStyle.theme()) do
        if options["analysis"] == "halogens"
            coefficients = halogen_coefficients(df; baseline=options["baseline"])
            CSV.write(joinpath(output, "halogen_coefficients.csv"), coefficients)
            MISTStyle.savefig("coefficients_electrochemical_1x3",
                coefficient_figure(coefficients, ["homo", "lumo", "gap"], (1, 3)); fig_dir=output)
            MISTStyle.savefig("coefficients_thermo_2x2",
                coefficient_figure(coefficients, ["bp", "mp", "fp", "Cv"], (2, 2)); fig_dir=output)
        else
            MISTStyle.savefig("Li_vs_SHE_and_O2_superoxide_vs_SHE_1x2", calibration_figure(df); fig_dir=output)
        end
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
