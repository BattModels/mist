using DataFrames
using StatsModels: FullDummyCoding
using GLM
using JSON
using CairoMakie

function get_duration(df, range)
    only(df[df.range  .== range, "duration"])
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
        nvtx = DataFrame(stats["nvtx"])
        max_memory = maximum(df -> maximum(df["memory_usage"]), values(stats["memory"])) / 1e9
        push!(rows, (;
            nodes = stats["config"]["nodes"],
            bucket = stats["config"]["deepspeed"]["zero_optimization"]["reduce_bucket_size"],
            stage = stats["config"]["deepspeed"]["zero_optimization"]["stage"],
            gas = stats["config"]["train"]["trainer.accumulate_grad_batches"],
            training_batch = get_duration(nvtx, "run_training_batch"),
            validation_batch = get_duration(nvtx, ".val_next"),
            training_dataloader = get_duration(nvtx, ".train_dataloader_next"),
            max_memory,
        ))
    end
    return DataFrame(rows)
end


function fit_models(df)
    contrasts = Dict(:stage => DummyCoding())
    # Training Batch
    lm(
        @formula( training_batch ~ 1 + stage + log2(gas)&stage + log10(bucket)&stage),
        df;
        contrasts
    ) |> display

    # Validation batch
    lm(
        @formula( validation_batch ~ 1 + stage + log2(gas)&stage + log10(bucket)&stage),
        df;
        contrasts
    ) |> display

    # Peak Memory Usage
    lm(
        @formula( max_memory ~ 1 + stage + log2(gas)&stage + log10(bucket)&stage),
        df;
        contrasts
    ) |> display
end
