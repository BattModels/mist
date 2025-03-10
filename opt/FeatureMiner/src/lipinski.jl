function load_fitted_probes(ckpt_dir)
    probes = []
    local ckpt_meta
    for ckpt in readdir(ckpt_dir; join=true)
        @info realpath(ckpt)
        if !startswith(basename(ckpt), "star")
            _, ckpt_meta = load_linear_probes(ckpt)
            continue
        end
        m = match(r"layer-(\d+)-(\w+).*?--auroc-([\d\.]+)\.ckpt", basename(ckpt))
        layer = parse(Int, m[1]) + 1
        location = m[2]
        auroc = parse(Float64, m[3])
        try
            ckpt_probes, ckpt_meta = load_linear_probes(ckpt)
            push!(probes, (;
                ckpt_probes[layer]...,
                location,
                auroc,
            ))
        catch e
            e isa InterruptException && rethrow()
            @error "failed to load $ckpt" e catch_backtrace()
            continue
        end
    end
    return DataFrame(probes), ckpt_meta
end

cosine_similarity(a::Vector, b::Vector) = dot(a, b) / (norm(a) * norm(b))

function additive_features(w::Matrix)
    f_overall = w[end, :]
    f_componets = vec(sum(w[1:end-1, :]; dims=1))
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

