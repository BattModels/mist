module TokenizerStats

using DataFrames
using Makie
using PythonCall
using ArgParse
using OnlineStats: OnlineStats, CountMap, HyperLogLog, Extrema, KHist, Counter, fit!, merge!, value
using LinearAlgebra: normalize
using StatsBase: StatsBase, Histogram, fit, AbstractWeights, nobs, mean, std
using MPI: MPI
using BSON: BSON
using JSON: JSON
using SHA: SHA
using SparseArrays: sparse

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

load_tokenizer(args...; kwargs...) = @pyconst(pyimport("electrolyte_fm.utils.tokenizer")).load_tokenizer(args...; kwargs...)
function split_dataset_by_node(args...; kwargs...)
    m = @pyconst(pyimport("datasets.distributed"))
    return m.split_dataset_by_node(args...; kwargs...)
end

function molnet_config(name::AbstractString)
    file = joinpath(@__DIR__, "..", "..", "..", "submit", "moleculenet_tasks.libsonnet")
    evaluate_file = @pyconst(pyimport("_jsonnet")).evaluate_file
    config = JSON.parse(pyconvert(String, evaluate_file(file)))
    return config[name]
end

function molnet(name::AbstractString; tokenizer="smirk", canonical::Bool=false)
    @assert !canonical "molnet datamodule doesn't support cannonical"
    config = molnet_config(name)
    data_modules = @pyconst(pyimport("electrolyte_fm.data_modules"))
    dm = data_modules.MolNetDataModule(name,
        tokenizer=tokenizer,
        target_columns=config["target_columns"],
        strip_unk_tokens=false,
        batch_size=1,
        val_batch_size=1,
        include_smiles=true,
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm
end

function pretrain(path::AbstractString; tokenizer="smirk", canonical::Bool=false)
    data_modules = @pyconst(pyimport("electrolyte_fm.data_modules"))
    dm = data_modules.RobertaDataSet(
        path,
        tokenizer,
        batch_size=1,
        val_batch_size=1,
        canonical
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm
end

include("ngrams.jl")
include("collect.jl")
include("plotting.jl")
include("finetune.jl")
include("cli.jl")

end
