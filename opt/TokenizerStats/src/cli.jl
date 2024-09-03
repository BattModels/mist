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
            default = abspath(joinpath(@__DIR__, "..", "stats"))
        "usage"
            help = "Tabule token usage statistics"
            action = :command
        "distortion"
            help = "Compute the information loss from unknown tokens"
            action = :command
        "loss"
            help = "Evaluate a ngram model on the train/validation set"
            action = :command
        "merge"
            help = "Merge n-gram counts from two datasets"
            action = :command
    end
    common_args!(s["usage"])
    @add_arg_table! s["usage"] begin
        "--splits"
            help = "Which splits to compute stats for (comman separated)"
            default = "train"
        "--mode"
            help = "Which mode to use for distributed computation"
            default = "mpi"
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
        "tokenizer"
            help = "Tokenizer to use, must match the ngram model"
            arg_type = String
            required = true
        "dataset"
            help = "Path to the dataset to process, or name of a MolNet Dataset"
            arg_type = String
            required = true
    end
    @add_arg_table! s["merge"] begin
        "--split"
            help = "Which splits to use for merging"
            default = "train"
            arg_type = String
        "tokenizer"
            help = "Path to folder containing the trained n-gram models"
            arg_type = String
            required = true
        "a"
            help = "Name of the first n-gram model"
            arg_type = String
            required = true
        "b"
            help = "Name of the second n-gram model"
            arg_type = String
            required = true
    end

    args = parse_args(args, s)
    isnothing(args) && return 0
    args_cmd = args[args["%COMMAND%"]]

    # Run command
    if args["%COMMAND%"] == "distortion"
        tokenizer = args_cmd["tokenizer"]
        tokenizer_name = isdir(tokenizer) ? basename(tokenizer) : tokenizer
        dm, dataset = get_dataset(args_cmd["dataset"], tokenizer)
        ref_name = BSON.load(args_cmd["reference"])[:tokenizer][:name]
        ref_name = replace(ref_name, "/" => "--")
        out_file = joinpath(args["output"], tokenizer_name, dataset, ref_name * "_info_loss.bson")
        avg_information_loss(dm, args_cmd["reference"], out_file)

    elseif args["%COMMAND%"] == "loss"
        # Load the tokenizer
        tokenizer = args_cmd["tokenizer"]
        tokenizer_name = isdir(tokenizer) ? basename(tokenizer) : tokenizer
        @assert BSON.load(args_cmd["ngram"])[:tokenizer][:name] == tokenizer_name "ngram model must use the same tokenizer"

        # Setup dataset
        dm, dataset = get_dataset(args_cmd["dataset"], tokenizer)
        ngram_name = first(splitext(basename(args_cmd["ngram"])))

        out_file = joinpath(args["output"], tokenizer_name, dataset,  ngram_name * "_model_loss.bson")
        model_loss(dm, args_cmd["ngram"], out_file)

    elseif args["%COMMAND%"] == "merge"
        tokenizer = args_cmd["tokenizer"]
        model_a = joinpath(tokenizer, args_cmd["a"] * ".bson")
        model_b = joinpath(tokenizer, args_cmd["b"] * ".bson")
        @assert isfile(model_a) && isfile(model_b) "Models must be saved to disk"
        output = joinpath(tokenizer, args_cmd["a"] * "_" * args_cmd["b"] * ".bson")
        merge_ngrams(model_a, model_b, output; split=args_cmd["split"])

    else
        tokenizer = args_cmd["tokenizer"]
        tokenizer_name = isdir(tokenizer) ? basename(tokenizer) : tokenizer
        dm, dataset = get_dataset(args_cmd["dataset"], tokenizer)
        out_file = joinpath(args["output"], tokenizer_name, dataset * ".bson")
        splits = split(args_cmd["splits"], ",")

        # Distribute computation
        if args_cmd["mode"] == "mpi"
            tabulate_dataset(dm, out_file; tokenizer_name, splits)
        elseif args_cmd["mode"] == "srun"
            @info "Using srun mode"
            srun_usage_stats(dm, out_file; tokenizer_name, splits)
        else
            error("Unknown mode $(args_cmd["mode"])")
        end
    end

    return 0
end

function merge_ngrams(a_file::String, b_file::String, output::String; split::String="train")
    # Load Models
    a = BSON.load(a_file)
    b = BSON.load(b_file)
    @assert a[:tokenizer] == b[:tokenizer] "N-gram models must use the same tokenizer"

    # Combine ngram counts from both models
    a_ngrams = _get_split(a, a_file, split)
    b_ngrams = _get_split(b, b_file, split)
    ngrams = map(zip(a_ngrams, b_ngrams)) do (a, b)
        mergewith!(+, a, b)
    end

    # Save merged model
    rm(output; force=true)
    BSON.bson(output;
        tokenizer = a[:tokenizer],
        samples = a[:samples] + b[:samples],
        ngrams,
    )
    chmod(output, 0o444)
    return nothing
end

function _get_split(data, file, split)
    if Symbol(split) ∉ keys(data) && :ngrams in keys(data)
        @warn "Using :ngram from $file for $split"
        return data[:ngrams]
    elseif Symbol(split) ∈ keys(data)
        return data[Symbol(split)]
    else
        error("No ngrams found for $split in $file")
    end
end
