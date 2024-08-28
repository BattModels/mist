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


function get_dataset(name_or_path, tokenizer)
    if isdir(name_or_path)
        dm = TokenizerStats.pretrain(name_or_path; tokenizer)
        dataset_name = basename(name_or_path)
    else
        dm = TokenizerStats.molnet(name_or_path; tokenizer)
        dataset_name = name_or_path
    end
    return dm, dataset_name
end

function main(args::Vector{String})
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--canonicalize"
            help = "Canonicalize the SMILES string before tokenizing"
            action = :store_true
        "--output"
            help = "Output stats directory"
            arg_type = String
            default = joinpath(@__DIR__, "..", "stats")
        "usage"
            help = "Tabule token usage statistics"
            action = :command
        "distortion"
            help = "Compute the information loss from unknown tokens"
            action = :command
        "loss"
            help = "Evaluate a ngram model on the train/validation set"
            action = :command
    end
    common_args!(s["usage"])
    @add_arg_table! s["usage"] begin
        "--splits"
            help = "Which splits to compute stats for (comman separated)"
            default = "train"
    end
    common_args!(s["distortion"])
    @add_arg_table! s["distortion"] begin
        "--reference", "-r"
            help = "Path to previously generated *.bson"
            arg_type = String
            required = true
    end
    @add_arg_table! s["loss"] begin
        "ngram"
            help = "Path to previously generated *.bson"
            arg_type = String
            required = true
        "dataset"
            help = "Path to the dataset to process, or name of a MolNet Dataset"
            arg_type = String
            required = true
    end

    args = parse_args(args, s)
    isnothing(args) && return 0
    args_cmd = args[args["%COMMAND%"]]

    # Run command
    args["output"] = realpath(args["output"])
    if args["%COMMAND%"] == "distortion"
        tokenizer = args_cmd["tokenizer"]
        tokenizer_name = isdir(tokenizer) ? basename(tokenizer) : tokenizer
        dm, dataset = get_dataset(args_cmd["dataset"], tokenizer)
        ref_name = BSON.load(args_cmd["reference"])[:tokenizer][:name]
        ref_name = replace(ref_name, "/" => "--")
        out_file = joinpath(args["output"], tokenizer_name, dataset *"_$(ref_name)_info_loss.bson")
        avg_information_loss(dm, args_cmd["reference"], out_file)

    elseif args["%COMMAND%"] == "loss"
        tokenizer = BSON.load(args_cmd["ngram"])[:tokenizer][:name]
        dm, dataset = get_dataset(args_cmd["dataset"], tokenizer)
        out_file = joinpath(args["output"], tokenizer, dataset * "_model_loss.bson")
        model_loss(dm, args_cmd["ngram"], out_file)

    else
        tokenizer = args_cmd["tokenizer"]
        tokenizer_name = isdir(tokenizer) ? basename(tokenizer) : tokenizer
        dm, dataset = get_dataset(args_cmd["dataset"], tokenizer)
        out_file = joinpath(args["output"], tokenizer_name, dataset * ".bson")
        tabulate_dataset(dm, out_file; tokenizer_name, splits=split(args_cmd["splits"], ","))
    end
    return 0
end
