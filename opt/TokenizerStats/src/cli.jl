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

function maybe_parse_env(T::Type, x::String)
    env = get(ENV, x, nothing)
    if !isnothing(env)
        return parse(T, env)
    end
    return parse(T, x)
end

@tracepoint function main(args::Vector{String})
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
        default = r"usage.+?_rank_\d+\.jld2$"
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
        ds = DatasetConfig(args_cmd["dataset"], args_cmd["tokenizer"], args_cmd["encoding"])
        avg_information_loss(ds, args_cmd["reference"], args_cmd["output"])

    elseif args["%COMMAND%"] == "loss"
        # Load the tokenizer
        ds = DatasetConfig(args_cmd["dataset"], args_cmd["tokenizer"], args_cmd["encoding"])
        model_loss(ds, args_cmd["model"], args_cmd["output"])

    elseif args["%COMMAND%"] == "merge"
        directory = args_cmd["directory"]
        pattern = args_cmd["pattern"]
        output = args_cmd["output"]
        files = find(directory, pattern)
        @info "Will merge $(length(files)) files into $output" files
        merge_usage_stats(files; output)

    elseif args["%COMMAND%"] == "usage"
        ds = DatasetConfig(args_cmd["dataset"], args_cmd["tokenizer"], args_cmd["encoding"])
        splits = parse_splits(args_cmd["splits"])

        # Distribute computation
        if args_cmd["mode"] == "mpi"
            tabulate_dataset(ds, args_cmd["output"], splits)
        elseif args_cmd["mode"] == "batch"
            world_size = maybe_parse_env(Int, args_cmd["size"])
            global_rank = maybe_parse_env(Int, args_cmd["rank"])
            @info "Using batch mode: $global_rank of $world_size (0-indexed)"
            job_array_usage_stats(ds, args_cmd["output"]; splits, world_size, global_rank)
        else
            error("Unknown mode $(args_cmd["mode"])")
        end
    end

    return 0
end

parse_splits(x::String) = parse_splits(split(x, ","))
function parse_splits(x::Vector{<:AbstractString})
    if length(x) == 1 && first(x) == "all"
        return ["val", "train", "test"]
    else
        return x
    end
end

function merge_usage_stats(files::Vector{String}; output::String="merged.jld2", splits::Vector{String}=["train", "val", "test"])
    merged_stats = Dict{String, Int}()
    rm(output * ".tmp", force=true)
    for file in files
        jldopen(output * ".tmp", "a+") do merged
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
                        merged_stats[joinpath(split, "out_of_vocab")] += other[split]["out_of_vocab"]
                        merged_stats[joinpath(split, "samples")] += other[split]["samples"]
                        mergewith!(+, merged[split]["fertility"], other[split]["fertility"])
                        mergewith!(+, merged[split]["nunique"], other[split]["nunique"])
                        for n in 1:length(merged[split]["ngrams"])
                            b_ngram = compact_ngrams(other[split]["ngrams"]["$n"])
                            mergewith!(+, merged[split]["ngrams"]["$n"], b_ngram)
                        end
                    else
                        merged_stats[joinpath(split, "out_of_vocab")] = other[split]["out_of_vocab"]
                        merged_stats[joinpath(split, "samples")] = other[split]["samples"]
                        merged[joinpath(split, "fertility")] = other[split]["fertility"]
                        merged[joinpath(split, "nunique")] = other[split]["nunique"]
                        for n in 1:length(other[split]["ngrams"])
                            merged[joinpath(split, "ngrams", string(n))] = compact_ngrams(other[split]["ngrams"]["$n"])
                        end
                    end
                end
            end
        end
    end
    # Write merged single-value stats
    jldopen(output * ".tmp", "a+") do merged
        for split in splits
            for k in ["out_of_vocab", "samples"]
                merged[joinpath(split, k)] = merged_stats[joinpath(split, k)]
            end
        end
    end

    # Finalize merged file
    mv(output * ".tmp", output; force=true)
    @info "saved results to $output" now()
    chmod(output, 0o444)

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
