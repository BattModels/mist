#!/usr/bin/env -S julia --project=@script --color=yes --startup-file=no
using DataFrames
using JLD2: JLD2
using FeatureMiner: FeatureMiner, load_fitted_probes, additive_features

function process_checkpoint(ckpt::String)
    ckpt_probes, ckpt_meta = load_fitted_probes(joinpath(ckpt, "checkpoints"))
    transform!(ckpt_probes,
        :weight => ByRow(additive_features) => :lipinski_alignment
    )
    probe_similarity = map(1:5) do idx
        FeatureMiner.layerwise_similarity(ckpt_probes.weight, idx)
    end
    probes = collect(zip(ckpt_probes.weight, ckpt_probes.bias))
    select!(ckpt_probes, Not(:weight))
    select!(ckpt_probes, Not(:bias))
    ckpt_meta = (; ckpt_meta..., id=basename(ckpt))
    return (;
        meta=ckpt_meta,
        probes,
        probe_stats=ckpt_probes,
        probe_similarity,
    )
end

function (@main)(args::Vector{String})
    ckpt = args[1]
    out = process_checkpoint(ckpt)
    JLD2.jldsave(joinpath(ckpt, "linear_probes.jld2"); out...)
end
