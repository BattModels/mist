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
    data = torch.load(ckpt; map_location=torch.device("cpu"))

    # Extract weights
    probe_weights = Dict()
    for (k, v) in data["state_dict"].items()
        if pyconvert(Bool, k.startswith("_probes"))
            probe_weights[pyconvert(String, k)] = pyconvert(Array, v)
        end
    end

    # Collate probes
    probes = []
    location = pyconvert(String, data["hyper_parameters"]["probes"]["init_args"]["location"])
    hidden_size = pyconvert(Int, data["hyper_parameters"]["probes"]["init_args"]["hidden_size"])
    for idx in range(0; length=fld(length(probe_weights), 2))
        push!(probes, (;
            weight=probe_weights["_probes.$idx.weight"],
            bias=probe_weights["_probes.$idx.bias"],
            location,
            hidden_size,
            layer=idx,
        ))
    end

    meta = (;
        name_or_path=pyconvert(String, data["hyper_parameters"]["model"]["init_args"]["name_or_path"]),
        dataset=pyconvert(String, data["datamodule_hyper_parameters"]["init_args"]["name_or_path"]),
        encoding=pyconvert(String, data["datamodule_hyper_parameters"]["init_args"]["encoding"]),
        tokenizer=pyconvert(String, data["datamodule_hyper_parameters"]["init_args"]["tokenizer"]),
    )

    return probes, meta
end

include("identification.jl")
include("stats.jl")
include("lipinski.jl")

end
