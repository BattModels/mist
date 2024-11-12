function common_args!(s)
    @add_arg_table! s begin
        "--encoding"
        help = "Encoding of the molecules"
        arg_type = String
        default = "smiles"
        "--output"
        help = "Path to output file"
        arg_type = String
        default = "-"
        "dataset"
        help = "Path to the dataset to process, or name of a MolNet Dataset"
        arg_type = String
        required = true
        "tokenizer"
        arg_type = String
        required = true
    end
end


function get_dataset(name_or_path, tokenizer, encoding)
    if isdir(name_or_path)
        if "tmQM" in splitpath(name_or_path)
            dm = TokenizerStats.tmqm(name_or_path; tokenizer, encoding)
            dataset_name = "tmQM"
        else
            dm = TokenizerStats.pretrain(name_or_path; tokenizer, encoding)
            dataset_name = basename(name_or_path)
        end
    else
        dm = TokenizerStats.molnet(name_or_path; tokenizer, encoding)
        dataset_name = name_or_path
    end
    return dm, dataset_name
end

function main(args::Vector{String})
    s = ArgParseSettings()
    @add_arg_table! s begin
        "usage"
        help = "Tabule token usage statistics"
        action = :command
        "distortion"
        help = "Compute the information loss from unknown tokens"
        action = :command
        "loss"
        help = "Evaluate a ngram model on the train/validation/test set"
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
    common_args!(s["loss"])
    @add_arg_table! s["loss"] begin
        "--model"
        help = "Path to previously generated n-gram model `*.bson`"
        arg_type = String
        required = true
    end
    @add_arg_table! s["merge"] begin
        "--split"
        help = "Which splits to use for merging"
        default = "train"
        arg_type = String
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
        dm, dataset = get_dataset(args_cmd["dataset"], tokenizer, args_cmd["encoding"])
        avg_information_loss(dm, args_cmd["reference"], args_cmd["output"])

    elseif args["%COMMAND%"] == "loss"
        # Load the tokenizer
        tokenizer = args_cmd["tokenizer"]
        tokenizer_name = isdir(tokenizer) ? basename(tokenizer) : tokenizer
        dm, dataset = get_dataset(args_cmd["dataset"], tokenizer, args_cmd["encoding"])
        if endswith(args_cmd["model"], ".bson")
            @assert BSON.load(args_cmd["model"])[:tokenizer][:name] == tokenizer_name "ngram model must use the same tokenizer"
        end
        model_loss(dm, args_cmd["model"], args_cmd["output"])

    elseif args["%COMMAND%"] == "merge"
        model_a = args_cmd["a"]
        model_b = args_cmd["b"]
        @assert isfile(model_a) && isfile(model_b) "Models must be saved to disk"
        merge_ngrams(model_a, model_b, args_cmd["output"]; split=args_cmd["split"])

    elseif args["%COMMAND%"] == "usage"
        tokenizer = args_cmd["tokenizer"]
        tokenizer_name = isdir(tokenizer) ? basename(tokenizer) : tokenizer
        dm, dataset = get_dataset(args_cmd["dataset"], tokenizer, args_cmd["encoding"])
        splits = split(args_cmd["splits"], ",")

        # Distribute computation
        if args_cmd["mode"] == "mpi"
            tabulate_dataset(dm, args_cmd["output"]; tokenizer_name, splits)
        elseif args_cmd["mode"] == "srun"
            @info "Using srun mode"
            srun_usage_stats(dm, args_cmd["output"]; tokenizer_name, splits)
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
        tokenizer=a[:tokenizer],
        samples=a[:samples] + b[:samples],
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
