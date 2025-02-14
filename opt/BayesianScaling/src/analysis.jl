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
    scaling = selectdim(chains, 3:scaling)
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
