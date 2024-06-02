#!/usr/bin/env -S julia --color=yes --project --threads=auto --startup-file=no
using SQLite
using DataFrames
using JSON

table_names(db) = map(Base.Fix2(getfield, :name), SQLite.tables(db))

function generate_sqlite(report)
    path, _ = splitext(report)
    sqlite_path =  path * ".sqlite"
    try
        db = SQLite.DB(sqlite_path)
        @assert "NVTX_EVENTS" in table_names(db)
        return sqlite_path
    catch
        run(Cmd(["nsys", "export", "--force-overwrite=true", "--quiet=true", "--type", "sqlite", "--output", sqlite_path, report]))
    end
    return sqlite_path
end

normalize_nvtx_event_name(::Missing) = missing
function normalize_nvtx_event_name(name::String)
    name = first(split(name, ",")) # Remove trailing tags
    name = last(split(name, "]"))  # Remove leading [pl]...
    return name
end

function rank_statistics(sqlite_path)
    db = SQLite.DB(sqlite_path)

    # NVTX Stats
    query = """
    SELECT
	avg(end - start)/1000000000 as duration,
	min(end - start)/1000000000 as min_duration,
	max(end - start)/1000000000 as max_duration,
	count(*) as instances,
    normalize_nvtx_event_name(text) as range
    FROM NVTX_EVENTS
    GROUP BY range
    ORDER BY duration DESC
    LIMIT 100
    """
    SQLite.@register db normalize_nvtx_event_name
    nvtx_stats = DBInterface.execute(db, query) |> DataFrame

    # Memory Profile
    if "CUDA_GPU_MEMORY_USAGE_EVENTS" in table_names(db)
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
    else
        memory_trace = missing
    end
    return (; nvtx_stats, memory_trace)
end

function job_config(output_dir)
    files = readdir(output_dir; join=true)
    output_path = filter(x -> endswith(x, ".out") || endswith(x, ".o$(basename(output_dir))"), files)
    m = open(first(output_path), "r") do fid
        match(r"^{$.*?^}$"sm, read(fid, String))
    end
    if !isnothing(m)
        return JSON.parse(m.match)
    else
        return Dict()
    end
end

function merge_duration(duration, instances)
    duration = sum(duration .* instances) / sum(instances)
    instances = sum(instances)
    return [(duration, instances)]
end

function merge_stats(profile_stats)
    stats = combine(groupby(vcat(profile_stats...), "range"),
                      "min_duration" => minimum,
                      "max_duration" => maximum,
                    ["duration", "instances"] => merge_duration => ["duration", "instances"],
                    nrow => "ranks";
        renamecols=false,
    )
    sort!(stats, "duration"; rev=true)
    return stats
end


function job_statistics(output_dir)
    files = readdir(output_dir; join=true)
    nsys_reports = filter(endswith(".nsys-rep"), files)
    profile_stats = Vector(undef, length(nsys_reports))
    memory_trace = similar(profile_stats)
    ranks = Vector{Int}(undef, length(nsys_reports))
    Threads.@threads :dynamic for (idx, file) in collect(enumerate(nsys_reports))
        sqlite_path = generate_sqlite(file)
        profile_stats[idx], memory_trace[idx] = rank_statistics(sqlite_path)
        ranks[idx] = parse(Int, split(splitext(basename(file))[1], "_")[4])
    end

    # Extract the job config from the input file
    config = job_config(output_dir)

    # Return as a dict
    return return Dict(
        "config" => config,
        "nvtx" => NamedTuple.(eachrow(merge_stats(profile_stats))),
        "memory" => Dict((rank => v for (rank, v) in zip(ranks, memory_trace))),
    )
end

!isinteractive() && JSON.print(job_statistics(first(ARGS)))
