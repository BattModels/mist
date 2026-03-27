using DataFrames
using Dates: Dates, DateTime
using JSON: JSON
using BayesianScaling: find, average_last

function dedup_trace(metric_traces, name::AbstractString)
    steps = metric_traces["step"]
    y = metric_traces[name]
    trace = Dict{eltype(steps), eltype(y)}()
    fn = x -> (!isnothing(x[2]) && !ismissing(x[2]))
    for (xt, yt) in Iterators.filter(fn, zip(steps, y))
        trace[xt] = yt isa Real ? yt : parse(Float64, yt)
    end
    x_out::Vector{Int} = (sort∘collect∘keys)(trace)
    y_out::Vector{Float64} = [trace[x] for x in x_out]
    return x_out, y_out
end

function get_metric(metrics::Vector; kwargs...)
    df = DataFrame(metrics)
    kwargs = map(kv -> kv[1] => ByRow(==(kv[2])), collect(kwargs))
    subset!(df, kwargs...)
    if nrow(df) == 1
        return first(df.value)
    elseif nrow(df) == 0
        return missing
    else
        error("Multiple matches: $df")
    end
end

function pretraining_runs(dir::AbstractString; smoothed_eval_batch=1e6)
    row = []
    for file in find(joinpath(dir, "pretraining"), r".*\.json")
        run = JSON.parsefile(file; null=missing, allownan=true)

        pretrained_models = [
            "electrolyte_fm.models.RoBERTa",
            "electrolyte_fm.models.RoBERTaPreLayerNorm",
        ]
        model_class = run["model"]["class_path"]
        ismissing(model_class) && continue
        model_class in pretrained_models || continue

        d_model = run["model"]["d_model"]
        ff_ratio = run["model"]["d_ff"] / d_model
        kv_size = fld(run["model"]["d_model"], run["model"]["n_heads"])
        aspect_ratio = d_model / run["model"]["n_layers"]
        beta = run["optimizer"]["betas"]
        beta1 = !ismissing(beta) ? beta[1] : missing
        beta2 = !ismissing(beta) ? beta[2] : missing

        # Average over the last 1e6 samples
        _, val_epoch_loss = dedup_trace(run["metric_traces"], "val_loss")
        effective_val_epoch = run["trainer"]["effective_val_epoch"]
        ismissing(effective_val_epoch) && continue
        val_loss_smooth = average_last(val_epoch_loss; n=smoothed_eval_batch / effective_val_epoch)

        # Get last/best metrics
        val_loss_best=get_metric(run["metrics"]; metric="loss_epoch", split="val", type="best")
        val_loss_min=get_metric(run["metrics"]; metric="loss_epoch", split="val", type="min")
        val_loss_last=get_metric(run["metrics"]; metric="loss_epoch", split="val", type="last")
        val_loss_best = ismissing(val_loss_best) ? val_loss_min : val_loss_best

        push!(row, (;
            id=run["id"],
            state=run["state"],
            user=run["user"],
            cluster=run["cluster"],
            commit=run["commit"],
            tags=string.(run["tags"]),
            created=DateTime(run["created"][1:23], Dates.ISODateTimeFormat),
            model_size=run["model"]["model_size"],
            model_class=run["model"]["class_path"],
            d_model,
            ff_ratio,
            kv_size,
            aspect_ratio,
            optimizer=run["optimizer"]["class_path"],
            tokenizer=run["data"]["tokenizer"],
            max_steps=run["trainer"]["num_training_steps"],
            step=get_metric(run["metrics"]; metric="global_step", type="last"),
            gas=run["trainer"]["gas"],
            effective_batch_size=run["trainer"]["effective_batch_size"],
            effective_val_epoch,
            lr=run["optimizer"]["lr"],
            beta1,
            beta2,
            val_loss_best,
            val_loss_last,
            val_loss_smooth
        ))
    end
    return DataFrame(row)
end
