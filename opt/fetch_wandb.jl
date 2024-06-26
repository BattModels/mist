using Wandb
using DataFrames
using PythonCall
using JSON
using ProgressBars

const Optional{T} = Union{T,Missing}

function model_size(d_model, d_ff, n_layers)
    attention_qkv = n_layers * 3 * d_model^2
    project = n_layers * d_model^2
    ff = n_layers = 2 * d_model * d_ff
    return attention_qkv + project + ff
end

function get_entry(config, item::String)
    pyisinstance(config, pytype(nothing)) && return missing
    pyisinstance(config, @py(str)) && return missing
    val = get(config, item, missing)
    !ismissing(val) && return val
    haskey(config, "init_args") && return get_entry(config["init_args"], item)
    return missing
end

function get_entry(T::Type, config, path::String...)
    local val
    try
        val = get_entry(config, path...)
    catch e
        @error lazy"Unable to get $path" config path e
        rethrow(e)
    end
    ismissing(val) && return missing
    return pyconvert(T, val)
end

function get_entry(config, path::String...)
    if haskey(config, first(path))
        root, rest... = path
        val = get_entry(config[root], rest...)
        !ismissing(val) && return val
    end

    # Check for an intermediate init_args
    if haskey(config, "init_args")
        val = get_entry(config["init_args"], path...)
        return val
    end
    # Couldn't find a matching path
    return missing
end

function get_cluster(row)
    row["hostname"] == "localhost" && return "h001"
    startswith(row["hostname"], "lh") && return "artemis"
    startswith(row["hostname"], "x") && return "polaris"
    return missing
end

function wandb_convert(T, v)
    if T <: Real
        v = @py float(v)
    end
    return pyconvert(T, v)
end
function get_trace(run, channels::Pair{String,Tuple{DataType,String}}...)
    wandb_channels = first.(channels)
    trace = run.scan_history(keys=pylist(wandb_channels))
    T = typejoin(map(x -> x[2][1], channels)...)
    results = Dict{String,Vector{T}}()
    for snapshot in trace
        for (channel, (T, sym)) in channels
            v = get(results, sym, T[])
            push!(v, wandb_convert(T, snapshot[channel]))
            results[sym] = v
        end
    end
    return isempty(results) ? missing : results
end

function run_summary(run)
    config = run.config
    @assert "val/loss_epoch" in run.summary
    row = Dict{String,Any}(
        "id" => pyconvert(String, run.id),
        "name" => pyconvert(String, run.name),
        "url" => pyconvert(String, run.url),
        "hostname" => pyconvert(String, run.metadata["host"]),
        "created" => pyconvert(String, run.metadata["startedAt"]),
        "d_ff" => get_entry(Int, config, "intermediate_size"),
        "d_model" => get_entry(Int, config, "hidden_size"),
        "n_layers" => get_entry(Int, config, "num_hidden_layers"),
        "n_heads" => get_entry(Int, config, "num_hidden_layers"),
        "lr" => get_entry(Float64, config, "cli", "model", "optimizer", "lr"),
        "tokenizer" => get_entry(String, config, "tokenizer"),
        "num_training_steps" => get_entry(Int, config, "cli", "model", "lr_schedule", "num_training_steps"),
        "steps" => pyconvert(Int, run.summary["trainer/global_step"]),
        "macro_batch_size" => get_entry(Int, config, "stats/train_macro_batch_size"),
        "dataset" => get_entry(String, config, "path"),
    )
    row["model_size"] = model_size(row["d_model"], row["d_ff"], row["n_layers"])
    row["cluster"] = get_cluster(row)
    row["dataset"] = ismissing(row["dataset"]) ? missing : last(splitpath(row["dataset"]))

    # Account for GAS
    gas = get_entry(Int, config, "cli", "trainer", "accumulate_grad_batches")
    row["gas"] = ismissing(gas) ? 1 : gas
    row["eff_batch_size"] = row["macro_batch_size"] * row["gas"]

    # Training Loss Trace
    row["train/loss_step"] = get_trace(run,
        "trainer/global_step" => (Int, "step"),
        "train/loss_step" => (Float64, "loss"),
    )
    row["min_train_loss"] = ismissing(row["train/loss_step"]) ? missing : minimum(row["train/loss_step"]["loss"])

    # Validation Loss Trace
    row["val/loss_epoch"] = get_trace(run,
        "trainer/global_step" => (Int, "step"),
        "val/loss_epoch" => (Float64, "loss"),
    )
    row["min_val_loss"] = ismissing(row["val/loss_epoch"]) ? missing : minimum(row["val/loss_epoch"]["loss"])

    # Learning Rate Trace
    lr_channel = filter(x -> startswith(x, "lr"), pyconvert(Vector{String}, runs[51].summary.keys()))
    if length(lr_channel) >= 1
        if length(lr_channel) > 1
            @warn "Multiple lr channels found, using first" lr_channel
        end
        row["lr_trace"] = get_trace(run,
            "trainer/global_step" => (Int, "step"),
            first(lr_channel) => (Float64, "lr"),
        )
    else
        row["lr_trace"] = missing
    end
    return row
end

function query_runs(; kwargs...)
    API = Wandb.wandb.Api()
    filters = @py {
        "State":{"\$in":["Crashed", "Finished"]},
        "tags":{"\$in":["pretraining"]},
        "summary_metrics.trainer/global_step":{"\$exists":true},
        "summary_metrics.val/loss_epoch":{"\$exists":true},
        "config.intermediate_size":{"\$exists":true},
        "config.hidden_size":{"\$exists":true},
        "config.num_hidden_layers":{"\$exists":true},
    }
    return collect(API.runs(path="incite-mist/mist"; filters, kwargs...))
end

function tabulate_runs(runs; cache=joinpath(@__DIR__, "..", ".cache", "run-export"))
    rows = []
    mkpath(cache)
    for run in ProgressBar(collect(runs))
        cache_path = joinpath(cache, pyconvert(String, run.id) * ".json")
        if isfile(cache_path)
            push!(rows, JSON.parsefile(cache_path; null=missing))
        else
            try
                row = run_summary(run)
                push!(rows, row)
                open(cache_path, "w") do fid
                    JSON.print(fid, row)
                end
            catch e
                if e isa AssertionError
                    @warn "Unable to use $(run.id)" e
                else
                    @error "unable to parse $(run.id)" e Base.catch_stack()
                    return run
                end
            end
        end
    end
    return DataFrame(rows)
end
