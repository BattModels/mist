#!/usr/bin/env -S julia --color=yes --startup-file=no --project=@script
# Command-line entrypoint for the existing mixture figure functions.

using CSV
using DataFrames
using JSON
using Makie
using MISTStyle
using Mixtures

module Panels
include("panel.jl")
end
module Parity
include("plot_excess.jl")
end
module DatasetPlots
include("dataset.jl")
end

function arguments(args)
    options = Dict{String,String}()
    iseven(length(args)) || error("Expected --option value pairs")
    allowed = ["analysis", "data-root", "dataset", "model", "output-dir", "label"]
    for index in 1:2:length(args)
        startswith(args[index], "--") || error("Expected an option: $(args[index])")
        key = args[index][3:end]
        key in allowed || error("Unknown option --$key")
        options[key] = args[index + 1]
    end
    for key in ["analysis", "output-dir"]
        haskey(options, key) || error("Missing --$key")
    end
    options["analysis"] in ["soap", "dataset", "parity", "outliers", "experimental", "thermal"] ||
        error("Unknown analysis: $(options["analysis"])")
    return options
end

function required_input(options, key; description=key)
    haskey(options, key) || error("Missing --$key ($description)")
    path = abspath(options[key])
    ispath(path) || error("Missing $description: $path. See opt/mixtures/README.md for the required paper inputs.")
    return path
end

function save_panel(name, figure, options)
    if haskey(options, "label")
        Label(figure[0, :], options["label"]; fontsize=7pt, tellwidth=false)
    end
    MISTStyle.savefig(name, figure; fig_dir=options["output-dir"])
end

function save_predictions(path, df)
    open(path, "w") do stream
        for row in eachrow(df)
            JSON.print(stream, Dict(name => row[name] for name in names(df)))
            println(stream)
        end
    end
end

function (@main)(args)
    options = arguments(args)
    analysis = options["analysis"]
    data_root = get(options, "data-root", joinpath(@__DIR__, "..", "..", "data"))
    files = Dict(
        "soap" => ["soap_similarity.csv"],
        "outliers" => ["soap_similarity.csv"],
        "experimental" => ["experimental_ACN_BC.csv", "experimental_ACN_EGDA.csv"],
        "thermal" => ["alpha_data.jsonc"],
    )
    for filename in get(files, analysis, String[])
        path = joinpath(data_root, "mixtures", filename)
        isfile(path) || error("Missing mixture input: $path. See opt/mixtures/README.md.")
    end
    dataset = analysis in ["parity", "dataset"] ? required_input(options, "dataset") : nothing
    model = analysis in ["parity", "outliers", "experimental", "thermal"] ?
        Mixtures.load_excess_model(required_input(options, "model"; description="local mixture model directory")) : nothing
    mkpath(options["output-dir"])
    with_theme(MISTStyle.theme()) do
        if analysis == "soap"
            figure = Figure(; size=(80mm, 65mm))
            Panels.plot_soap!(figure[1, 1]; data_dir=data_root)
            save_panel("soap-correlation", figure, options)
        elseif analysis == "dataset"
            df = Mixtures.normalize_dataset(DataFrame(CSV.File(dataset)))
            save_panel("mixture-dataset", DatasetPlots.mixture_dataset(df), options)
        elseif analysis == "parity"
            # Preserve evaluate_dataset's validation-split convention from the
            # original figure code; do not choose checkpoints on test data.
            df = Mixtures.evaluate_dataset(model, dataset)
            save_predictions(joinpath(options["output-dir"], "predictions.jsonl"), df)
            extrema = Mixtures.excess_skew(df)
            CSV.write(joinpath(options["output-dir"], "composition-extrema.csv"), extrema;
                transform=(column, value) -> value isa AbstractVector ? JSON.json(value) : value)
            save_panel("excess-parity", Parity.parity_plots(df, model), options)
            figure = Figure(; size=(80mm, 65mm))
            Panels.plot_excess_skewness!(figure[1, 1], df)
            save_panel("composition-extrema", figure, options)
        else
            figure = Figure(; size=(80mm, 65mm))
            if analysis == "outliers"
                Panels.plot_soap_outliers!(figure[1, 1], model; data_dir=data_root)
                save_panel("soap-outlier-curves", figure, options)
            elseif analysis == "experimental"
                Panels.plot_experimental_data!(figure[1, 1], model; data_dir=data_root)
                save_panel("experimental-repeats", figure, options)
            else
                Panels.plot_thermal_alpha!(figure[1, 1], model; data_dir=data_root)
                save_panel("thermal-expansion", figure, options)
            end
        end
    end
    return 0
end
