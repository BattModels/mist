function collate_performance_stats(sweep::String)
    rows = []
    for run in readdir(sweep)
        config_file = joinpath(sweep, run, "config.json")
        logfile = joinpath(sweep, run, "screen.jsonl")
        (isfile(config_file) && isfile(logfile)) || continue

        config = JSON.parsefile(config_file)
        trace, rank_throughput = performance_trace(logfile)
        isempty(trace) && continue
        push!(rows, (;
            id=basename(run),
            path=joinpath(sweep, run),
            config,
            batch_size=config["generation"]["batch_size"],
            limit_db_fragments=config["generation"]["limit_db_fragments"],
            limit_ref_fragments=config["generation"]["limit_ref_fragments"],
            n_fragments=config["generation"]["n_fragments"],
            epoch_size=config["generation"]["epoch_size"] / config["generation"]["n_fragments"],
            unique_molecules=trace[end].unique_molecules,
            n_passing=trace[end].n_passing,
            duration=trace[end].time,
            gpus=length(rank_throughput),
            trace,
            rank_throughput=collect(values(rank_throughput)),
        ))
    end
    return DataFrame(rows)
end

function performance_trace(logfile::String)
    trace = []
    rank_throughput = Dict{Int,Float64}()
    for line in eachline(logfile)
        msg = JSON.parse(line)
        if haskey(msg, "n_passing_world") && get(msg, "global_rank", -1) == 0
            # Track global generation stats
            push!(trace, (;
                time=msg["elapsed_perf"],
                n_passing=msg["n_passing_world"],
                unique_molecules=msg["unique_molecules_world"],
            ))
        elseif haskey(msg, "passing_rank")
            # Record the final throughput of each rank
            rank = msg["global_rank"]
            rank_throughput[rank] = msg["eval_throughput_rank"]
        end
    end
    return trace, rank_throughput
end

