using Wandb
using DataFrames
using PythonCall
using JSON
using ProgressBars

const Optional{T} = Union{T,Missing} where {T}

function model_size(d_model, d_ff, n_layers)
    attention_qkv = n_layers * 3 * d_model^2
    project = n_layers * d_model^2
    ff = n_layers = 2 * d_model * d_ff
    return attention_qkv + project + ff
end

function get_entry(config, item::String)
    val = get(config, item, missing)
    !ismissing(val) && return val
    haskey(config, "init_args") && return get_entry(config["init_args"], item)
    return missing
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

function run_summary(run)
    config = run.config
    @assert pyisinstance(get_entry(config, "cli", "model", "optimizer"), pytype(PyDict())) "missing optimizer config"
    @assert pyisinstance(get_entry(config, "cli", "model", "lr_schedule"), pytype(PyDict())) "missing lr schedule"
    @assert "val/loss_epoch" in run.summary
    row = Dict{String,Any}(
        "id" => pyconvert(String, run.id),
        "name" => pyconvert(String, run.name),
        "url" => pyconvert(String, run.url),
        "created" => pyconvert(String, run.metadata["startedAt"]),
        "d_ff" => pyconvert(Int, get_entry(config, "intermediate_size")),
        "d_model" => pyconvert(Int, get_entry(config, "hidden_size")),
        "n_layers" => pyconvert(Int, get_entry(config, "num_hidden_layers")),
        "n_heads" => pyconvert(Int, get_entry(config, "num_hidden_layers")),
        "lr" => pyconvert(Float64, get_entry(config, "cli", "model", "optimizer", "lr")),
        "tokenizer" => pyconvert(String, get_entry(config, "tokenizer")),
        "num_training_steps" => pyconvert(Int, get_entry(config, "cli", "model", "lr_schedule", "num_training_steps")),
        "steps" => pyconvert(Int, run.summary["trainer/global_step"]),
        "macro_batch_size" => pyconvert(Int, get_entry(config, "stats/train_macro_batch_size")),
        "min_val_loss" => pyconvert(Float64, run.summary["val/loss_epoch"]["min"]),
    )
    row["model_size"] = model_size(row["d_model"], row["d_ff"], row["n_layers"])

    # Account for GAS
    gas = pyconvert(Optional{Int}, get_entry(config, "cli", "trainer", "accumulate_grad_batches"))
    row["eff_batch_size"] = row["macro_batch_size"] * ( ismissing(gas) ? 1 : gas )

    # Get Loss Trace
    loss_trace = run.scan_history(keys=@py(["trainer/global_step", "val/loss_epoch"]))
    row["loss"] = map(loss_trace) do snapshot
        (;
            step=pyconvert(Int, snapshot["trainer/global_step"]),
            loss=pyconvert(Float64, snapshot["val/loss_epoch"]),
        )
    end
    return row
end

function query_runs(; kwargs...)
    API = Wandb.wandb.Api()
    filters = @py {
        "State": {"\$in": ["Crashed", "Finished"]},
        "tags":{"\$in":["pretraining"]},
        "summary_metrics.trainer/global_step": {"\$exists": true},
        # "summary_metrics.trainer/global_step": {"\$gt": 2000},
        "config.intermediate_size": {"\$exists": true},
        "config.hidden_size": {"\$exists": true},
        "config.num_hidden_layers": {"\$exists": true},
        "summary_metrics.val/loss_epoch.min": {"\$gte": 1e-3},
    }
    return collect(API.runs(path="incite-mist/mist"; filters, kwargs...))
end

function tabulate_runs(runs; cache=joinpath(@__DIR__, "..", ".cache", "run-export"))
    rows = []
    mkpath(cache)
    for run in ProgressBar(collect(runs))
        cache_path = joinpath(cache, pyconvert(String, run.id) * ".json")
        if isfile(cache_path)
            push!(rows, JSON.parsefile(cache_path))
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
                end
            end
        end
    end
    return DataFrame(rows)
end
