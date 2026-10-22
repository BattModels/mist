#!/usr/bin/env -S julia --startup-file=no --project=@script
# Render the original tmQM application figures from released validation predictions.

using DataFrames
using JSON
using MISTStyle

include("tmqm.jl")

const PAPER_IDS = ["LEYLAC", "LIMTIK", "ROMVAQ", "ROMVEU"]

function arguments(args)
    iseven(length(args)) || error("Expected --predictions FILE --output-dir DIRECTORY")
    options = Dict{String, String}()
    for index in 1:2:length(args)
        args[index] in ["--predictions", "--output-dir"] || error("Unknown option $(args[index])")
        options[args[index][3:end]] = args[index + 1]
    end
    all(haskey(options, key) for key in ["predictions", "output-dir"]) ||
        error("Supply --predictions and --output-dir")
    return options
end

function paper_frames(path)
    isfile(path) || error("Missing original tmQM validation predictions: $path. Run tmqm-paper-validation-predictions.")
    df = load_jsonl(path)
    required = ["smiles", "id", "Metal_q", "Metal_q_mist", "HL_Gap", "HL_Gap_mist", "Polarizability", "Polarizability_mist"]
    all(name in names(df) for name in required) || error("Predictions require columns: $(join(required, ", "))")
    unique!(df, :id)
    all(id in df.id for id in PAPER_IDS) || error("Missing the paper's four annotated stereoisomers: $(join(PAPER_IDS, ", "))")
    pairs = find_chiral_pairs(df)
    all(id in pairs.id for id in PAPER_IDS) || error("The annotated stereoisomer pairs were lost during RDKit grouping")
    parity = select(
        pairs, :id => :cid, :smiles, :isomeric_smiles,
        :Metal_q => :metal_q_ref, :Metal_q_mist => :metal_q_mist,
        :HL_Gap => :gap_ref, :HL_Gap_mist => :gap_mist
    )
    return df, pairs, parity
end

function write_rows(path, frame)
    return open(path, "w") do stream
        for row in eachrow(frame)
            JSON.print(stream, Dict(name => row[name] for name in names(frame)))
            println(stream)
        end
    end
end

function (@main)(args)
    options = arguments(args)
    df, pairs, parity = paper_frames(options["predictions"])
    output = options["output-dir"]
    mkpath(output)
    # Save values in their original units before figure_stero converts gaps to eV.
    write_rows(joinpath(output, "stereoisomer-pairs.jsonl"), parity)
    write_rows(joinpath(output, "annotated-examples.jsonl"), subset(parity, :cid => ByRow(∈(PAPER_IDS))))
    MISTStyle.savefig("tmQM", figure_stero(copy(parity)); fig_dir = output)
    MISTStyle.savefig("tmQM-isomer-gaps", figure_chn(df); fig_dir = output)
    return MISTStyle.savefig("tmQM-complexes", figure_complexes(df, pairs); fig_dir = output)
end
