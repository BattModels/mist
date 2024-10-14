function find(dir, pattern)
    found = String[]
    for (root, dirs, files) in walkdir(dir)
        for file in files
            path = joinpath(root, file)
            if match(pattern, path) !== nothing
                push!(found, path)
            end
        end
    end
    return found
end

function pretraining_runs(dir::AbstractString=WANDB_EXPORT_DIR)
    row = []
    for file in find(joinpath(dir, "pretraining"), r".*\.json")
        run = JSON.parsefile(file; null=missing)
        d_model=run["model"]["d_model"]
        ff_ratio=run["model"]["d_ff"] / d_model
        kv_size= Int(run["model"]["d_model"] // run["model"]["n_heads"])
        aspect_ratio = d_model / run["model"]["n_layers"]
        beta = run["optimizer"]["betas"]
        beta1 = !ismissing(beta) ? beta[1] : missing
        beta2 = !ismissing(beta) ? beta[2] : missing
        push!(row, (;
            id=run["id"],
            state=run["state"],
            user=run["user"],
            cluster=run["cluster"],
            commit=run["commit"],
            tags=string.(run["tags"]),
            created=DateTime(run["created"][1:23], Dates.ISODateTimeFormat),
            model_size=run["model"]["model_size"],
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
            lr=run["optimizer"]["lr"],
            beta1,
            beta2,
            val_loss_best=run["metrics"]["val_loss_best"],
            val_loss_last=run["metrics"]["val_loss_last"],
        ))
    end
    return DataFrame(row)
end
