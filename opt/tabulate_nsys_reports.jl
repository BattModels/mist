#!/usr/bin/env -S julia --color=auto --project --threads=auto --startup-file=no
using SQLite
using DataFrames
using JSON
using CSV
using Statistics

const Optional{T} = Union{Missing,T}

table_names(db) = map(Base.Fix2(getfield, :name), SQLite.tables(db))

function generate_sqlite(report)
    path, _ = splitext(report)
    sqlite_path = path * ".sqlite"
    cmd = Cmd(["nsys",
        "export",
        "--force-overwrite=true",
        "--type=sqlite",
        "--quiet=true",
        "--output", sqlite_path,
        report,
    ])
    run(Cmd(cmd; ignorestatus=false))
    return sqlite_path
end

""" Returns a path to a valid sqlite report or missing if the report is invalid """
function get_valid_db(report)
    dbpath = report * ".sqlite"
    if !isfile(dbpath)
        dbpath = generate_sqlite(report)
    end

    # Check if report is valid
    if !("NVTX_EVENTS" in table_names(SQLite.DB(dbpath)))
        @warn lazy"Failed to generate sqlite file for $report"
        invalid = mkpath(joinpath(dirname(report), "invalid/"))
        rm(dbpath; force=true)
        mv(report, joinpath(invalid, basename(report)); force=true)
        return missing
    end
    return dbpath
end

normalize_nvtx_event_name(::Missing) = missing
function normalize_nvtx_event_name(name::String)
    name = first(split(name, ",")) # Remove trailing tags
    name = last(split(name, "]"))  # Remove leading [pl]...
    return name
end

function nvtx_stats(db)
    query = """
    SELECT
    avg(end - start)/1000000000 as duration,
    min(end - start)/1000000000 as min_duration,
    max(end - start)/1000000000 as max_duration,
    min(start)/1000000000 as first_start,
    count(*) as instances,
    normalize_nvtx_event_name(text) as range
    FROM NVTX_EVENTS
    GROUP BY range
    ORDER BY duration DESC
    LIMIT 100
    """
    SQLite.@register db normalize_nvtx_event_name
    return eachrow(DataFrame(DBInterface.execute(db, query)))
end

function memory_usage(db)
    query = """
    SELECT
        ROUND(start/1000000000, 1)  as time,
        SUM(CASE WHEN memoryOperationType == 0 THEN bytes ELSE -bytes END) as bytes,
        deviceId
    FROM
    CUDA_GPU_MEMORY_USAGE_EVENTS
    GROUP BY time, deviceId
    ORDER BY time
    """
    memory_trace = DBInterface.execute(db, query) |> DataFrame
    transform!(groupby(memory_trace, "deviceId"),
        "bytes" => cumsum => "memory_usage",
    )
    return eachrow(memory_trace)
end

function extract_config(logfile)
    m = open(logfile, "r") do fid
        match(r"^{$.*?^}$"sm, read(fid, String))
    end
    isnothing(m) && return missing
    return JSON.parse(m.match)
end

function job_statistics(output_dir; limit=nothing)
    files = readdir(output_dir; join=true)
    nsys_reports = filter(endswith(".nsys-rep"), files)
    if !isnothing(limit)
        nsys_reports = nsys_reports[1:limit]
    end
    isempty(nsys_reports) && return missing

    # Check if benchmarking completed
    logfile = filter(files) do file
        !isnothing(match(r"\.out$", file)) && return true
        !isnothing(match(r"^.*?\.o\d*$", file)) && return true
        return false
    end |> only
    if false && isnothing(match(r"`Trainer\.fit` stopped:", read(logfile, String)))
        @info lazy"Benchmarking did not complete, skipping $output_dir"
        return nothing
    end

    # Extract the job config
    config = extract_config(logfile)

    # Extract benchmarking stats
    rank_statistics = Dict{Int,Optional{Dict}}()
    l = ReentrantLock()
    Threads.@threads :dynamic for file in nsys_reports
        @info "Processing $file"
        dbpath = get_valid_db(file)
        rank = parse(Int, split(splitext(basename(file))[1], "_")[4])
        if ismissing(dbpath)
            @lock l rank_statistics[rank] = missing
        else
            db = SQLite.DB(dbpath)
            nvtx = nvtx_stats(db)
            if "CUDA_GPU_MEMORY_USAGE_EVENTS" in table_names(db)
                memory = memory_usage(db)
            else
                memory = missing
            end
            stats = Dict("nvtx" => nvtx, "memory" => memory)
            @lock l rank_statistics[rank] = stats
        end
    end

    # Return as a dict
    return Dict("config" => config, "ranks" => rank_statistics)
end

function process_sweep(run_directory; kwargs...)
    Threads.@threads :dynamic for dir in readdir(run_directory; join=true)
        process_dir(dir)
    end
end

function process_dir(dir)
    @info "Processing $dir"
    stats = job_statistics(dir)
    isnothing(stats) && return
    open(joinpath(dir, "stats.json"), "w") do fid
        JSON.print(fid, stats)
    end
end

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
        isnothing(v["memory"]) && continue
        push!(usage, f(Base.Fix2(getindex, "memory_usage"), v["memory"]))
    end
    isempty(usage) && return missing
    return merge(stack(usage)) ./ 1e9
end

function maybe_get(stats::Dict, path::String...)
    for key in path
        stats = get(stats, key, missing)
        ismissing(stats) && return missing
    end
    return stats
end

"""
Collate tabulated results into a single dataframe
"""
function collate_results(sweep_dir)
    rows = []
    for file in readlines(`find $sweep_dir -name 'stats.json'`)
        stats = JSON.parsefile(file)
        if isnothing(stats) || !haskey(stats, "ranks")
            @warn "No ranks found in $file"
            continue
        end
        rank_stats = stats["ranks"]
        max_memory = get_max_memory(rank_stats)
        avg_rank_max_memory = get_max_memory(rank_stats; merge=mean)

        push!(rows, (;
            nodes=maybe_get(stats, "config", "nodes"),
            gpus_per_node=maybe_get(stats, "config", "gpus_per_node"),
            bucket=maybe_get(stats, "config", "deepspeed", "zero_optimization", "reduce_bucket_size"),
            stage=maybe_get(stats, "config", "deepspeed", "zero_optimization", "stage"),
            gas=maybe_get(stats, "config", "train", "trainer.accumulate_grad_batches"),
            batch_size=maybe_get(stats, "config", "train", "data.batch_size"),
            d_model=maybe_get(stats, "config", "train", "model.hidden_size"),
            d_ff=maybe_get(stats, "config", "train", "model.intermediate_size"),
            n_layers=maybe_get(stats, "config", "train", "model.num_hidden_layers"),
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

"""
Script for processing nsys output files

    $(basename(@__FILE__)) [dir]                            # Process a single directory
    $(basename(@__FILE__)) sweep [dir]                      # Process a directory of results
    $(basename(@__FILE__)) collate [dir] > summary.csv      # Collate

## Multi-threaded   
Script is multi-threaded and defaults to automatically picking the number of threads.
However, when running in a batch job, setting the `JULIA_NUM_THREADS` directly is recommended.

## Example SBATCH
```bash
#!/bin/bash
#SBATCH -c 16

module load cuda/12.3   # CUDA >= 12.2 is required
export JULIA_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-auto}
./opt/$(basename(@__FILE__)) RESULT_DIR # Or similar
```

## Installation
Before running you will need to install julia and instantiate the environment

```bash
# Install Julia with Juliaup
curl -fsSL https://install.julialang.org | sh

# Install the environment (Run from the project root)
julia --project -e 'using Pkg; Pkg.instantiate()'

# Run the script
./opt/$(basename(@__FILE__)) RESULT_DIR # Or similar
```

"""
function main(args)
    if length(args) == 1
        process_dir(args[1])
        return 0
    elseif length(args) == 2 && args[1] == "sweep"
        process_sweep(args[2])
        return 0
    elseif length(args) == 2 && args[1] == "collate"
        df = collate_results(args[2])
        CSV.write(stdout, df)
        return 0
    else
        display(@doc(main))
        return 1
    end
end

!isinteractive() && exit(main(ARGS))
