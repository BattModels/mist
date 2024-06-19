using DataFrames
using JSON
using CairoMakie

function get_duration(df, range)
    only(df[df.range.==range, "duration"])
end

function merge_stat(merge::Function, f::Function, ranks::Dict, stat::String, range::String)
    results = []
    for v in values(ranks)
        ismissing(v) && continue
        idx = findall(==(range), v["range"])
        !isempty(idx) && push!(results, f(v[stat][idx]))
    end
    if isempty(results)
        return missing
    else
        return merge(stack(results))
    end
end

function weighted_duration(ranks::Dict, range)
    duration = merge_stat(identity, identity, ranks, "duration", range)
    instances = merge_stat(identity, identity, ranks, "instances", range)
    (ismissing(duration) || ismissing(instances)) && return missing
    return sum(duration .* instances) / sum(instances)
end

function get_max_memory(stats)
    !haskey(stats, "memory") && return missing
    usage = Int[]
    for v in values(stats["memory"])
        isnothing(v) && continue
        if haskey(v, "memory_usage")
            push!(usage, maximum(v["memory_usage"]))
        end
    end
    isempty(usage) && return missing
    return maximum(usage) / 1e9
end

"""
Collate NVTX results produced by `opt/job_stats` for sweep over
deepspeed config setting for a single model
"""
function collate_results(sweep_dir)
    rows = []
    for dir in readdir(sweep_dir; join=true)
        file = joinpath(dir, "stats.json")
        if !isfile(file)
            @warn "Missing results for $dir"
            continue
        end
        stats = JSON.parsefile(file)
        nvtx = stats["nvtx"]
        max_memory = get_max_memory(stats)

        push!(rows, (;
            nodes=stats["config"]["nodes"],
            bucket=stats["config"]["deepspeed"]["zero_optimization"]["reduce_bucket_size"],
            stage=stats["config"]["deepspeed"]["zero_optimization"]["stage"],
            gas=stats["config"]["train"]["trainer.accumulate_grad_batches"],
            startup_time=merge_stat(maximum, minimum, nvtx, "first_start", "run_training_batch"),
            training_batch=weighted_duration(nvtx, "run_training_batch"),
            validation_batch=weighted_duration(nvtx, ".val_next"),
            training_dataloader=weighted_duration(nvtx, ".train_dataloader_next"),
            max_memory,
        ))
    end
    return DataFrame(rows)
end


# function fit_models(df)
#     contrasts = Dict(:stage => DummyCoding())
#     # Training Batch
#     lm(
#         @formula(training_batch ~ 1 + stage + log2(gas) & stage + log10(bucket) & stage),
#         df;
#         contrasts
#     ) |> display
#
#     # Validation batch
#     lm(
#         @formula(validation_batch ~ 1 + stage + log2(gas) & stage + log10(bucket) & stage),
#         df;
#         contrasts
#     ) |> display
#
#     # Peak Memory Usage
#     lm(
#         @formula(max_memory ~ 1 + stage + log2(gas) & stage + log10(bucket) & stage),
#         df;
#         contrasts
#     ) |> display
# end
