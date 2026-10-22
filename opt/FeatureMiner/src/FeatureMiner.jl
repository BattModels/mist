module FeatureMiner

using ArgParse
using DataFrames
using PythonCall: Py, pyimport, pyconvert, @pyconst, GIL
using OnlineStats: OnlineStats, KHist, Variance, Series, fit!
using OnlineStatsBase: OnlineStatsBase, OnlineStat, EqualWeight, smooth, bessel, nobs
using StatsBase: StatsBase
using LinearAlgebra: norm, dot

function FeatureExtractor(ckpt_path::String)
    cls = pyimport("electrolyte_fm.models.sae.FeatureExtractor")
    return cls.from_checkpoint(ckpt_path)
end

function FeaturizedSmiles(path::String, miner::Py)
    cls = pyimport("electrolyte_fm.data_modules.sae_dataset.FeaturizedSmiles")
    return cls(path, miner)
end

function split_dataset_by_node(dataset::Py, rank::Int, size::Int)
    m = pyimport("datasets.distributed")
    return m.split_dataset_by_node(dataset, rank, size)
end

function load_linear_probes(ckpt)
    torch = @pyconst(pyimport("torch"))
    # These are our own local Lightning checkpoints, including their metadata.
    data = torch.load(ckpt; map_location="cpu", weights_only=false)

    # Extract weights
    probe_weights = Dict()
    for (k, v) in data["state_dict"].items()
        if pyconvert(Bool, k.startswith("_probes"))
            probe_weights[pyconvert(String, k)] = pyconvert(Array, v.detach().float().numpy())
        end
    end

    # Collate probes
    probes = []
    names = pyconvert(Vector{String}, data["state_dict"].get("_prob_points", []))
    metrics = data.get("probe_metrics", Dict())
    for idx in range(0; length=fld(length(probe_weights), 2))
        if isempty(names)
            # Older per-layer checkpoints did not record their hook names.
            location = pyconvert(String, data["hyper_parameters"]["probes"]["init_args"]["location"])
            name = "*encoder.layer.$idx.$location"
        else
            name = names[idx + 1]
        end
        hook = match(r"\.layer\.(\d+)\.(.+)$", name)
        isnothing(hook) && error("Unrecognized probe hook: $name")
        # Put the overall Lipinski rule first, as required by additive_features.
        order = [5, 1, 2, 3, 4]
        weight = probe_weights["_probes.$idx.weight"][order, :]
        push!(probes, (;
            weight,
            bias=probe_weights["_probes.$idx.bias"][order],
            location=String(hook[2]),
            hidden_size=size(weight, 2),
            layer=parse(Int, hook[1]) + 1,
            val_loss=pyconvert(Float64, metrics.get("val/$name-loss", NaN)),
            auroc=pyconvert(Float64, metrics.get("val/$name-auroc", NaN)),
            epoch=pyconvert(Int, data["epoch"]),
            selected_step=pyconvert(Int, data["global_step"]),
            checkpoint=abspath(ckpt),
        ))
    end

    model_args = data["hyper_parameters"].get("model", Dict()).get("init_args", Dict())
    data_meta = data.get("datamodule_hyper_parameters", Dict())
    data_args = data_meta.get("init_args", data_meta)
    meta = (;
        name_or_path=pyconvert(String, model_args.get("name_or_path", model_args.get("encoder", ""))),
        dataset=pyconvert(String, data_args.get("path", data_args.get("name_or_path", ""))),
        encoding=pyconvert(String, data_args.get("encoding", "smiles-kekule")),
        tokenizer=pyconvert(String, data_args.get("tokenizer", "")),
    )

    return probes, meta
end

include("identification.jl")
include("stats.jl")
include("lipinski.jl")

end
