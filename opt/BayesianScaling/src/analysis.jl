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

function design(m, chains; d_model=768, ff_ratio=4, n_layers=8, kv_size=64, gpus=32, gas=16, batch_size=128)
    N = non_embedding_size(d_model, ff_ratio * d_model, n_layers)
    scaling = selectdim(chains, 3, :scaling)
    A = selectdim(scaling, 3, :A)
    α = selectdim(scaling, 3, :α)
    B = selectdim(scaling, 3, :B)
    β = selectdim(scaling, 3, :β)

    # Compute-optimal budget
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    C = @. 6 * (N / G)^(1 / a)
    D = @. C / (6 * N)
    effective_batch_size = batch_size * gas * gpus
    S = D ./ effective_batch_size

    p = 0.9
    lr_dist = credible_interval(p)
    loss_dist = credible_interval(p)
    S_dist = credible_interval(p)
    C_dist = credible_interval(p)
    aspect_ratio = d_model / n_layers
    effective_batch_size = batch_size * gas * gpus
    for I in CartesianIndices(axes(chains)[1:2])
        θ = chains[I.I..., :]

        # Compute-optimal budget
        A = θ.scaling.A
        α = θ.scaling.α
        B = θ.scaling.B
        β = θ.scaling.β
        G = ((α * A) / (β * B))^(1 / (α + β))
        a = β / (α + β)
        C = 6 * (N / G)^(1 / a)
        D = C / (6 * N)
        S = D / effective_batch_size
        fit!(S_dist, S)
        fit!(C_dist, C)

        lr = ideal_lr(m, θ; model_size=N, effective_batch_size)
        fit!(lr_dist, lr)

        # Estimate the model's loss
        run = (;
            model_size=N,
            data_size=D,
            lr,
            effective_batch_size,
            ff_ratio,
            aspect_ratio,
            kv_size,
        )
        loss = expected_loss(m, θ, run)
        fit!(loss_dist, loss)
    end

    println("Non-Embedding Parameters: $N")
    println("\td_model: $d_model")
    println("\td_ff: $(d_model * ff_ratio)")
    println("\tn_attn $(Int(d_model / kv_size))")
    println("\tn_layers: $n_layers")
    println("\tkv_size: $kv_size")
    report_dist("PF-Days", C_dist; factor=inv(pf_day))
    report_dist("Number of Steps", S_dist)
    report_dist("Number of Samples", S_dist; factor=effective_batch_size)
    report_dist("Learning Rate", lr_dist)
    report_dist("Loss", loss_dist)
end

function report_dist(label::AbstractString, dist::OnlineStat; p=0.025, factor=1)
    m, l, u = OnlineStats.value(dist)
    lr = round(l * factor, sigdigits=3)
    m = round(m * factor, sigdigits=3)
    ur = round(u * factor, sigdigits=3)
    if lr == ur
        range = round((u - l) / 2, sigdigits=3)
        uq = "±$range"
    else
        uq = " [$lr, $ur]"
    end

    println("$(label): $m$uq")
end
