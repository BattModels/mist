function common_args!(s)
    @add_arg_table! s begin
        "dataset"
            help = "Path to the dataset to process, or name of a MolNet Dataset"
            arg_type = String
            required = true
        "tokenizer"
            arg_type = String
            required = true
    end
end

function main(args::Vector{String})
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--canonicalize"
            help = "Canonicalize the SMILES string before tokenizing"
            action = :store_true
        "--output"
            help = "Path of output file, will generate if unset"
            arg_type = String
        "usage"
            help = "Tabule token usage statistics"
            action = :command
        "distortion"
            help = "Compute the information loss from unknown tokens"
            action = :command
    end
    common_args!(s["usage"])
    common_args!(s["distortion"])
    @add_arg_table! s["distortion"] begin
        "--reference", "-r"
            help = "Path to previously generated *.bson"
            arg_type = String
            required = true
    end
    args = parse_args(s)
    args_cmd = args[args["%COMMAND%"]]

    if isdir(args_cmd["dataset"])
        dm = TokenizerStats.pretrain(args_cmd["dataset"]; tokenizer=args_cmd["tokenizer"])
        dataset_name = basename(args_cmd["dataset"])
    else
        dm = TokenizerStats.molnet(args_cmd["dataset"]; tokenizer=args_cmd["tokenizer"])
        dataset_name = args_cmd["dataset"]
    end

    if isdir(args_cmd["tokenizer"])
        tokenizer_name = basename(args_cmd["tokenizer"])
    else
        tokenizer_name = args_cmd["tokenizer"]
    end

    # Parse output file
    if isnothing(args["output"])
        out_file = joinpath(@__DIR__, "stats", args_cmd["tokenizer"], dataset_name)
    else
        out_file = args["output"]
    end
    out_file = out_file * ".bson"
    @info "will save results to $out_file"

    if args["%COMMAND%"] == "distortion"
        TokenizerStats.avg_information_loss(dm, args_cmd["reference"], dirname(out_file))
    else
        tabulate_dataset(dm, out_file; tokenizer_name)
    end
    return 0
end
