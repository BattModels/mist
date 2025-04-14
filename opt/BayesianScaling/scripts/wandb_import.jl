using DataFrames
using Dates: Dates, DateTime
using JSON: JSON
using BayesianScaling: find, average_last

function pretraining_runs(dir::AbstractString; smoothed_eval_batch=1e6)
    row = []
    for file in find(joinpath(dir, "pretraining"), r".*\.json")
        run = JSON.parsefile(file; null=missing)

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
        val_epoch_loss = map(run["metric_traces"]["val_loss"]) do x
            x isa Real ? x : parse(Float64, x)
        end
        effective_val_epoch = run["trainer"]["effective_val_epoch"]
        ismissing(effective_val_epoch) && continue
        val_loss_smooth = average_last(val_epoch_loss; n=smoothed_eval_batch / effective_val_epoch)

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
            step=run["trainer"]["step"],
            tokens=run["trainer"]["tokens"],
            gas=run["trainer"]["gas"],
            masked_tokens=run["trainer"]["masked_tokens"],
            effective_batch_size=run["trainer"]["effective_batch_size"],
            effective_val_epoch,
            lr=run["optimizer"]["lr"],
            beta1,
            beta2,
            val_loss_best=run["metrics"]["val_loss_best"],
            val_loss_last=run["metrics"]["val_loss_last"],
            val_loss_smooth
        ))
    end
    return DataFrame(row)
end
