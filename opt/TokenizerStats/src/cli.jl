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
    merged = jldopen(output, "w")
    for file in files
        jldopen(file, "r") do other
            @info "Merging $file" other
            if haskey(other, "tokenizer")
                if haskey(merged, "tokenizer")
                    @assert other["tokenizer"] == merged["tokenizer"] "N-gram models must use the same tokenizer"
                else
                    merged["tokenizer"] = other["tokenizer"]
                end
            end

            for split in splits
                if haskey(merged, split)
                    merged[split]["out_of_vocab"] += other["out_of_vocab"]
                    merged[split]["samples"] += other["samples"]
                    merged[split]["out_of_vocab"] += other["out_of_vocab"]
                    merged[split]["fertility"] = Dict(mergewith(+, merged[split]["fertility"], other["fertility"]))
                    merged[split]["nunique"] = Dict(mergewith(+, merged[split]["nunique"], other["nunique"]))
                    for n in 1:length(merged[split]["ngrams"])
                        a_ngram = merged[split]["ngrams"]["$n"]
                        b_ngram = compact_ngrams(other["ngrams"]["$n"])
                        merged[split]["ngrams"]["$n"] = Dict(mergewith(+, a_ngram, b_ngram))
                    end
                else
                    merged[split]["out_of_vocab"] = other[split]["out_of_vocab"]
                    merged[split]["samples"] = other[split]["samples"]
                    merged[split]["out_of_vocab"] = other[split]["out_of_vocab"]
                    merged[split]["fertility"] = other[split]["fertility"]
                    merged[split]["nunique"] = other[split]["nunique"]
                    for n in 1:length(other[split]["ngrams"])
                        merged[split]["ngrams"]["$n"] = compact_ngrams(other[split]["ngrams"]["$n"])
                    end
                end
            end
        end
    end
    return output
end

"""
When merging multi-file usage stats, switch counts to Float32 (avoid overflow) and
ids to UInt16 (reduce memory usage)

> Some tokenizer (e.g. Llama) use more tokens than UInt16 can hold, but for the
> tokenizers requiring merging (APE and SmilesPE) this is fine.

> TODO: Dynamically set this based on vocab size
"""
function compact_ngrams(T::Type, ngrams::AbstractDict)
    Dict(T.(k) => Float32(v) for (k, v) in pairs(ngrams))
end
compact_ngrams(ngrams::AbstractDict) = compact_ngrams(UInt16, ngrams)
