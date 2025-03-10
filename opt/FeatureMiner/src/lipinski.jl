function load_fitted_probes(ckpt_dir)
    probes = []
    for ckpt in readdir(ckpt_dir; join=true)
        startswith(basename(ckpt), "star") || continue
        m = match(r"layer-(\d+)-(\w+).*?--auroc-([\d\.]+)\.ckpt", basename(ckpt))
        layer = parse(Int, m[1]) + 1
        location = m[2]
        auroc = parse(Float64, m[3])
        ckpt_probes, ckpt_meta = load_linear_probes(ckpt)
        push!(probes, (;
            ckpt_probes[layer]...,
            model=ckpt_meta.name_or_path,
            auroc,
        ))
    end
    return DataFrame(probes)
end

function additive_features(w::Matrix)
    f_overall = w[end, :]
    f_componets = sum(w[1:end-1, :]; dims=1)
    return dot(f_overall, f_componets) / (norm(f_overall) * norm(f_componets))
end
