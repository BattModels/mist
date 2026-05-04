#!/usr/bin/env -S julia +release --color=auto --startup-file=no --project=@script
# Script to dump the neural scaling campaign to a latex table
using PrettyTables: LatexTableFormat, LatexCell, pretty_table
using DataFrames
using JLD2: jldopen
using Format: format
using BayesianScaling: BayesianScaling, observations, response

include("utils.jl")

function format_table(df; sigdigits=3)
    columns = [
        "model_size" => "Model Size",
        "data_size" => "Data Size",
        "effective_batch_size" => "Effective Batch Size",
        "ff_ratio" => "FF Ratio",
        "aspect_ratio" => "Aspect Ratio",
        "kv_size" => "KVQ Size",
        "lr" => "Learning Rate",
        "loss" => "Validation Loss",
    ]
    df = select(df, first.(columns))
    sort!(df, first.(columns))
    rename!(df, columns...)

    function fmt_lr(v, i, j)
        col = first(columns[j])
        if col == "lr"
            return LatexCell(sn(v; sigdigits))
        elseif col == "ff_ratio"
            return format("{:d}", Int(v))
        elseif col == "aspect_ratio"
            return format("{:.2f}", v)
        elseif col in ["model_size", "data_size"]
            return format(Int(v); commas=true)
        elseif col == "loss"
            return LatexCell(sn(v; sigdigits))
        else
            return v
        end
    end

    tf = LatexTableFormat(;
        header_envs=[],
        subheader_envs=[],
    )

    return pretty_table(df;
        tf,
        formatters=(fmt_lr,),
        alignment=:c,
        backend=Val(:latex),
        hlines=[:header],
        header=last.(columns),
        table_type=:longtable,
    )
end

function (@main)(ARGS=[])
    data = jldopen(ARGS[1], "r")
    model = data["model"]
    df = DataFrame(observations(model))
    df.loss = response(model)

    # Total FLOPs to train models
    # Average smirk fertility per Wadell, A. et al. 2025. Tokenization for Molecular Foundation Models.
    tokens_per_molecule = 69.2
    @info "Total Compute" sum(6.0 .* df.model_size .* df.data_size .* tokens_per_molecule)

    print(format_table(df))
end
