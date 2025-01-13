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


@annotate function get_dataset(name_or_path, tokenizer, encoding)
    start = time()
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
    @info "loaded $dataset_name in $(time() - start) s"
    return dm, dataset_name
end

function maybe_parse_env(T::Type, x::String)
    env = get(ENV, x, nothing)
    if !isnothing(env)
        return parse(T, env)
    end
    return parse(T, x)
end

@annotate function main(args::Vector{String})
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
        "--size"
        help = "Number of nodes to use for distributed computation, only used with --mode=batch. Can be an environment variable."
        default = "SLURM_ARRAY_TASK_COUNT"
        "--rank"
        help = "Rank of the task, only used with --mode=batch. Can be an environment variable."
        default = "SLURM_ARRAY_TASK_ID"
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
        "--pattern"
        default = r"usage.+?_rank_\d+\.jld2"
        arg_type = Regex
        "directory"
        help = "Directory to search for files to merge"
        arg_type = String
        "output"
        default = "merged.jld2"
        arg_type = String
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
        directory = args_cmd["directory"]
        pattern = args_cmd["pattern"]
        output = args_cmd["output"]
        files = find(directory, pattern)
        @info "Will merge $(length(files)) files into $output" files
        merge_usage_stats(files; output)

    elseif args["%COMMAND%"] == "usage"
        tokenizer = args_cmd["tokenizer"]
        tokenizer_name = isdir(tokenizer) ? basename(tokenizer) : tokenizer
        dm, dataset = get_dataset(args_cmd["dataset"], tokenizer, args_cmd["encoding"])
        splits = split(args_cmd["splits"], ",")

        # Distribute computation
        if args_cmd["mode"] == "mpi"
            tabulate_dataset(dm, args_cmd["output"]; tokenizer_name, splits)
        elseif args_cmd["mode"] == "batch"
            size = maybe_parse_env(Int, args_cmd["size"])
            rank = maybe_parse_env(Int, args_cmd["rank"])
            @info "Using batch mode: $rank of $size (0-indexed)"
            job_array_usage_stats(dm, args_cmd["output"]; tokenizer_name, splits, size, rank)
        else
            error("Unknown mode $(args_cmd["mode"])")
        end
    end

    return 0
end

function merge_usage_stats(files::Vector{String}; output::String="merged.jld2", splits::Vector{String}=["train", "val", "test"])
    data = Dict{String,Any}()
    for file in files
        jldopen(file, "r") do other
            @info "Merging $file" other
            if haskey(other, "tokenizer")
                if haskey(data, "tokenizer")
                    @assert other["tokenizer"] == data["tokenizer"] "N-gram models must use the same tokenizer"
                else
                    data["tokenizer"] = other["tokenizer"]
                end
            end

            for split in splits
                if haskey(data, split)
                    data[split] = _merge_usage_stats(data[split], other[split])
                else
                    data[split] = other[split]
                end
            end
        end
    end
    jldopen(output, "w") do f
        for (k, v) in data
            f[k] = v
        end
    end
    return output
end

function _merge_usage_stats(a::NamedTuple, b::NamedTuple)
    @assert Set(keys(a)) == Set(keys(b)) == Set([:samples, keys(tracked_stats())...])
    samples = a.samples + b.samples
    nunique = mergewith(+, a.nunique, b.nunique)
    fertility = mergewith(+, a.fertility, b.fertility)
    out_of_vocab = a.out_of_vocab + b.out_of_vocab
    ngrams = map(1:5) do n
        ang = a.ngrams[n]
        bng = b.ngrams[n]
        @assert keytype(ang) <: NTuple{n}
        @assert keytype(bng) <: NTuple{n}
        mergewith(+, compact_ngrams(ang), compact_ngrams(bng))
    end
    out = (; samples, nunique, fertility, out_of_vocab, ngrams)
    @assert Set(keys(out)) == Set([:samples, keys(tracked_stats())...])
    return out
end

function compact_ngrams(ngrams::AbstractDict)
    keytype(ngrams) == UInt16 && valuetype(ngrams) == Float32 && return ngrams
    map(collect(pairs(ngrams))) do (k, v)
        k = UInt16.(k)
        k => Float32(v)
    end |> Dict
end
