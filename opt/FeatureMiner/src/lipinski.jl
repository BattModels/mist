function load_fitted_probes(ckpt_dir)
    probes = []
    ckpt_meta = nothing
    for ckpt in readdir(ckpt_dir; join=true)
        endswith(ckpt, ".ckpt") || continue
        startswith(basename(ckpt), "star") || continue
        m = match(r"layer-(\d+)-([\w\-]+?)--.*?--loss-([-+\d.eE]+)--auroc-([-+\d.eE]+)(?:-v\d+)?\.ckpt$", basename(ckpt))
        isnothing(m) && error("Unrecognized probe checkpoint: $ckpt")
        layer = parse(Int, m[1]) + 1
        ckpt_probes, ckpt_meta = load_linear_probes(ckpt)
        # All-hook checkpoints contain several probes per layer. Match the
        # selected hook, not its layer number as an index into that list.
        probe = only(filter(p -> p.layer == layer && replace(p.location, "." => "-") == m[2], ckpt_probes))
        push!(probes, (;
            probe...,
            val_loss=isfinite(probe.val_loss) ? probe.val_loss : parse(Float64, m[3]),
            auroc=isfinite(probe.auroc) ? probe.auroc : parse(Float64, m[4]),
        ))
    end
    isempty(probes) && error("No fitted probes found in $ckpt_dir")
    df = DataFrame(probes)
    summary_path = joinpath(dirname(ckpt_dir), "fit.pt")
    if isfile(summary_path)
        summary = pyimport("torch").load(summary_path; map_location="cpu", weights_only=true)
        df.training_steps .= pyconvert(Int, summary["training_steps"])
        df.requested_steps .= pyconvert(Int, summary["requested_steps"])
        df.step_budget_reached .= pyconvert(Bool, summary["step_budget_reached"])
    end
    sort!(df, [:location, :layer])
    return df, ckpt_meta
end

cosine_similarity(a::Vector, b::Vector) = dot(a, b) / (norm(a) * norm(b))

function additive_features(w::Matrix)
    f_overall = w[1, :]
    f_componets = vec(sum(w[2:end, :]; dims=1))
    return cosine_similarity(f_overall, f_componets)
end

function layerwise_similarity(weights::Vector{W}, feature::Int) where {W<:Matrix{T}} where {T}
    N = length(weights)
    s = Matrix{T}(undef, N, N)
    for I in CartesianIndices(s)
        s[I] = cosine_similarity(weights[I[1]][feature, :], weights[I[2]][feature, :])
    end
    return s
end
