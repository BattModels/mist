#!/usr/bin/env -S julia --project --startup-file=no
using TokenizerStats: tabulate_dataset
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
            default = "stats.json"
        "dataset"
            help = "Path to the dataset to process"
            arg_type = String
        "tokenizer"
            arg_type = String
    end
    args = parse_args(s)
    out_file = length(args) == 3 ? args[3] : "stats.json"
    tabulate_dataset(args["dataset"], args["tokenizer"], args["output"]; canonical=args["canonicalize"])
end

!isinteractive() && main(ARGS)
