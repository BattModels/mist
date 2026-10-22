#!/usr/bin/env -S julia --color=yes --startup-file=no --project=@script
# CLI for the original ionic-conductivity plots and their descriptor tables.

using CSV
using DataFrames
using GLM
using JSON
using Makie
using MISTStyle
using Mixtures

module Ionic
include("ionic_conductivity.jl")
end
module Panels
include("panel.jl")
end

function arguments(args)
    options = Dict{String,String}()
    iseven(length(args)) || error("Expected --option value pairs")
    allowed = ["analysis", "model", "output-dir", "data-csv", "similarity", "parameter",
               "solvent", "salt", "excess", "molarity", "label"]
    for index in 1:2:length(args)
        startswith(args[index], "--") || error("Expected an option: $(args[index])")
        key = args[index][3:end]
        key in allowed || error("Unknown option --$key")
        options[key] = args[index + 1]
    end
    for key in ["analysis", "output-dir"]
        haskey(options, key) || error("Missing --$key")
    end
    options["analysis"] in ["tables", "main", "non-arrhenius", "angell", "composition", "delta-ea", "ternary", "soap", "molarity"] ||
        error("Unknown ionic analysis: $(options["analysis"])")
    return options
end

function required_input(options, key)
    haskey(options, key) || error("Missing --$key")
    path = abspath(options[key])
    ispath(path) || error("Missing ionic-conductivity input: $path. See opt/mixtures/README.md for its source and schema.")
    return path
end

function save_panel(name, figure, options)
    if haskey(options, "label")
        Label(figure[-1, :], options["label"]; fontsize=7pt, tellwidth=false)
    end
    MISTStyle.savefig(name, figure; fig_dir=options["output-dir"])
end

function molarity_panels(model, molarity, options)
    fit_figure = Figure(; size=(100mm, 50mm))
    descriptor_figure = Figure(; size=(120mm, 55mm))
    descriptor_axes = [
        Axis(descriptor_figure[1, 1]; xlabel="Melting Point (K)", ylabel=L"T_0\;(K)", xautolimitmargin=(0.06, 0.18)),
        Axis(descriptor_figure[1, 2]; xlabel=L"\varepsilon", ylabel=L"T_0\;(K)", xautolimitmargin=(0.06, 0.18)),
    ]
    tables = DataFrame[]
    coefficients = DataFrame[]
    for (index, salt) in enumerate(Ionic.salts)
        # Keep the original model-derived T0 values and the original linear fit.
        fitted, df = Ionic.fit_T0_dielectric_model(model, salt, molarity)
        Ionic.plot_T0_model_fit!(fit_figure[1, index], fitted, df, salt)
        salt_name = "Li" * Ionic.label_smi(salt)
        for (axis, column) in zip(descriptor_axes, [:melting_point, :dielectric])
            scatter!(axis, df[!, column], df.T0; label=salt_name, color=MISTStyle.CAT_COLORS[index], marker=:circle)
            text!(axis, df[!, column], df.T0; text=df.solvent, fontsize=4pt,
                  align=(:left, :center), offset=(2, 0))
        end
        df.salt = fill(salt_name, nrow(df))
        df.molarity = fill(molarity, nrow(df))
        push!(tables, df)
        push!(coefficients, DataFrame(salt=salt_name, term=coefnames(fitted),
                                      coefficient=coef(fitted), standard_error=stderror(fitted)))
    end
    colgap!(fit_figure.layout, 9)
    rowgap!(fit_figure.layout, 0)
    axislegend(descriptor_axes[2]; position=:lt)
    CSV.write(joinpath(options["output-dir"], "unary-properties.csv"), vcat(tables...))
    CSV.write(joinpath(options["output-dir"], "descriptor-coefficients.csv"), vcat(coefficients...))
    save_panel("gml_fit_$(molarity)M", fit_figure, options)
    save_panel("descriptors_vs_T0_$(molarity)M", descriptor_figure, options)
end

function (@main)(args)
    options = arguments(args)
    analysis = options["analysis"]
    csv = analysis in ["main", "non-arrhenius", "angell"] ? required_input(options, "data-csv") : nothing
    similarity = analysis == "soap" ? JSON.parsefile(required_input(options, "similarity")) : nothing
    model = isnothing(csv) ? Mixtures.load_conductivity_model(required_input(options, "model")) : nothing
    mkpath(options["output-dir"])
    if analysis == "tables"
        curves, inference = Ionic.conductivity_tables(model)
        CSV.write(joinpath(options["output-dir"], "ionic_conductivity_curves.csv"), curves)
        CSV.write(joinpath(options["output-dir"], "ionic_conductivity_inference.csv"), inference)
        return 0
    end
    with_theme(MISTStyle.theme()) do
        if analysis == "main"
            figure = Figure(; size=(61mm, 45mm))
            Panels.plot_ionic_conductivity!(figure[1, 1]; data_path=csv, generated=true)
            save_panel("ionic_prediction", figure, options)
        elseif analysis == "non-arrhenius"
            df = DataFrame(CSV.File(csv))
            subset!(df, "x5 (cation, mole fraction)" => ByRow(>(0.14)))
            figure = Figure(; size=(90mm, 80mm), figure_padding=(2, 2, 2, 2))
            Ionic.plot_non_arr_data!(figure, df)
            save_panel("non_arr_data", figure, options)
        elseif analysis == "angell"
            save_panel("angell-solvents", Ionic.plot_angell_solvents(; data_path=csv, generated=true), options)
        elseif analysis == "composition"
            for solvent in Ionic.solvents
                name = "cond_curve_" * join(Ionic.label_smi.(solvent), "_")
                save_panel(name, Ionic.plot_composition_curves(model, solvent), options)
            end
        elseif analysis == "delta-ea"
            save_panel("delta_Ea", Ionic.plot_delta_Ea(model), options)
        elseif analysis == "ternary"
            parameter = get(options, "parameter", "Ea")
            parameter in ["Ea", "Tg"] || error("--parameter must be Ea or Tg (T0)")
            solvent = options["solvent"]
            salt = options["salt"]
            solvent in ["FEC", "EMC", "DEC", "DMC"] || error("Unknown ternary solvent: $solvent")
            salt in ["PF6", "TFSI"] || error("Unknown salt: $salt")
            excess = parse(Bool, get(options, "excess", "false"))
            mixture = Dict(
                "solvents" => ["CC1COC(=O)O1", "O=C1OCCO1", Ionic.label_smi(solvent; input="acr")],
                "temperature" => 298.15,
                "salt" => [Ionic.label_smi(salt; input="acr"), "[Li+]"],
            )
            name = (excess ? "excess_" : "") * "$(parameter)_$(solvent)_$(salt)"
            save_panel(name, Ionic.plot_ternary_vft_parameter(model, mixture; parameter, excess), options)
        elseif analysis == "soap"
            figure = Figure(; size=(90mm, 120mm), figure_padding=(2, 2, 2, 2))
            Ionic.plot_soap_similarity_correlation!(figure, model, 298; similarity_data=similarity)
            save_panel("soap_corr_T_298", figure, options)
        else
            molarity = parse(Float64, options["molarity"])
            molarity > 0 || error("Molarity must be positive")
            molarity_panels(model, molarity, options)
        end
    end
    return 0
end
