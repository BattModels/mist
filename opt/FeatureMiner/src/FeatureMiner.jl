module FeatureMiner

using ArgParse
using PythonCall: Py, pyimport
using OnlineStats: OnlineStats, KHist, Variance, Series, fit!
using OnlineStatsBase: OnlineStatsBase, OnlineStat, EqualWeight, smooth, bessel, nobs
using StatsBase: StatsBase


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

include("identification.jl")
include("stats.jl")

end

