using DataFrames
using JSON
using CairoMakie
using Statistics

function merge_stat(merge::Function, f::Function, ranks::Dict, stat::String, range::String; group="nvtx")
    results = []
    for v in values(ranks)
        ismissing(v) && continue
        !haskey(v, group) && continue
        for item in v[group]
            if item["range"] == range
                push!(results, f(item[stat]))
            end
        end
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

function get_max_memory(rank_stats, group="memory"; merge=maximum, f=maximum)
    usage = Int[]
    for v in values(rank_stats)
        !haskey(v, group) && continue
        push!(usage, f(Base.Fix2(getindex, "memory_usage"), v["memory"]))
    end
    isempty(usage) && return missing
    return merge(stack(usage)) ./ 1e9
end

"""
Collate NVTX results produced by `opt/job_stats` for sweep over
deepspeed config setting for a single model
"""
function collate_results(sweep_dir)
    rows = []
    for file in readlines(`find $sweep_dir -name 'stats.json'`)
        @info file
        stats = JSON.parsefile(file)
        rank_stats = stats["ranks"]
        max_memory = get_max_memory(rank_stats)
        avg_rank_max_memory = get_max_memory(rank_stats; merge=mean)

        push!(rows, (;
            nodes=stats["config"]["nodes"],
            bucket=stats["config"]["deepspeed"]["zero_optimization"]["reduce_bucket_size"],
            stage=stats["config"]["deepspeed"]["zero_optimization"]["stage"],
            gas=stats["config"]["train"]["trainer.accumulate_grad_batches"],
            startup_time=merge_stat(maximum, minimum, rank_stats, "first_start", "run_training_batch"),
            training_batch=weighted_duration(rank_stats, "run_training_batch"),
            validation_batch=weighted_duration(rank_stats, ".val_next"),
            training_dataloader=weighted_duration(rank_stats, ".train_dataloader_next"),
            max_memory,
            avg_rank_max_memory,
        ))
    end
    return DataFrame(rows)
end

