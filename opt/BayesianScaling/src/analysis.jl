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
        d_model = run["model"]["d_model"]
        ff_ratio = run["model"]["d_ff"] / d_model
        kv_size = fld(run["model"]["d_model"], run["model"]["n_heads"])
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

function design(m, chains; d_model=768, ff_ratio=4, n_layers=8, kv_size=64, gpus=32, gas=16, batch_size=128)
    N = non_embedding_size(d_model, ff_ratio * d_model, n_layers)
    scaling = chains[:, :, :scaling]
    A = scaling[:, :, :A]
    α = scaling[:, :, :α]
    B = scaling[:, :, :B]
    β = scaling[:, :, :β]

    # Compute-optimal budget
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    C = @. 6 * (N / G)^(1 / a)
    D = @. C / (6 * N)
    effective_batch_size = batch_size * gas * gpus
    S = D ./ effective_batch_size

    lr = ideal_lr(m, chains; model_size=N, effective_batch_size)

    loss = Array{Float64}(undef, size(chains)[1:2]...)
    aspect_ratio = d_model / n_layers
    for sdx in CartesianIndices(axes(loss))
        run = (; model_size=N, data_size=rand(D), lr=rand(lr), effective_batch_size, ff_ratio, aspect_ratio, kv_size)
        loss[sdx] = expected_loss(m, chains[sdx.I..., :], run)
    end

    println("Non-Embedding Parameters: $N")
    println("\td_model: $d_model")
    println("\td_ff: $(d_model * ff_ratio)")
    println("\tn_attn $(Int(d_model / kv_size))")
    println("\tn_layers: $n_layers")
    println("\tkv_size: $kv_size")
    report_dist("FLOPS", vec(C))
    report_dist("Number of Steps", vec(S))
    report_dist("Number of Samples", vec(D))
    report_dist("Learning Rate", vec(lr))
    report_dist("Loss", vec(loss))
end

function report_dist(label::AbstractString, dist::AbstractVector; p=0.025)
    l, m, u = quantile(dist, [p, 0.5, 1 - p])
    lr = round(l, sigdigits=3)
    m = round(m, sigdigits=3)
    ur = round(u, sigdigits=3)
    if lr == ur
        range = round((u - l) / 2, sigdigits=3)
        uq = "±$range"
    else
        uq = " [$lr, $ur]"
    end

    println("$(label): $m$uq")
end
