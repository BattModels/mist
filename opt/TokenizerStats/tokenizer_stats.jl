#!/usr/bin/env -S julia --project --startup-file=no
using TokenizerStats: TokenizerStats, tabulate_dataset
using ArgParse

function main(args::Vector{String})
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--canonicalize"
            help = "Canonicalize the SMILES string before tokenizing"
            action = :store_true
        "--output"
            help = "Path of output file"
            arg_type = String
        "dataset"
            help = "Path to the dataset to process, or name of a MolNet Dataset"
            arg_type = String
            required = true
        "tokenizer"
            arg_type = String
            required = true
    end
    args = parse_args(s)
    if isdir(args["dataset"])
        dm = TokenizerStats.pretrain(args["dataset"]; tokenizer=args["tokenizer"])
        dataset_name = basename(args["dataset"])
    else
        dm = TokenizerStats.molnet(args["dataset"]; tokenizer=args["tokenizer"])
        dataset_name = args["dataset"]
    end

    # Parse output file
    if isnothing(args["output"])
        out_file = joinpath(@__DIR__, "stats", args["tokenizer"], dataset_name)
    else
        out_file = args["output"]
    end
    out_file = out_file * ".json"
    @info "will save results to $out_file"


    tabulate_dataset(dm, out_file; tokenizer_name=args["tokenizer"])
end

!isinteractive() && main(ARGS)
