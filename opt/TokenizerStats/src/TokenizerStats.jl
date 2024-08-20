module TokenizerStats

using DataFrames
using Makie
using PythonCall
using OnlineStats: OnlineStats, CountMap, HyperLogLog, Extrema, KHist, Counter, fit!, merge!, value
using LinearAlgebra: normalize
using StatsBase: StatsBase, Histogram, fit, AbstractWeights, nobs
using MPI: MPI
using JSON: JSON

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

# Delay loading python code until initialization
# https://juliapy.github.io/PythonCall.jl/stable/pythoncall/#Precompilation
const PY_DISTRIBUTED = Ref{Py}()
const PY_DATAMODULE = Ref{Py}()
const PY_JSONNET = Ref{Py}()

split_dataset_by_node(args...; kwargs...) = PY_DISTRIBUTED[].split_dataset_by_node(args...; kwargs...)

function molnet_config(name::AbstractString)
    file = joinpath(@__DIR__, "..", "..", "..", "submit", "moleculenet_tasks.libsonnet")
    config = JSON.parse(pyconvert(String, PY_JSONNET[].evaluate_file(file)))
    return config[name]
end

function molnet(name::AbstractString; tokenizer="smirk", canonical::Bool=false)
    @assert !canonical "molnet datamodule doesn't support cannonical"
    config = molnet_config(name)
    dm = PY_DATAMODULE[].MolNetDataModule(name,
        tokenizer=tokenizer,
        target_columns=config["target_columns"],
        strip_unk_tokens=false,
        batch_size=1,
        val_batch_size=1,
    )
    dm.prepare_data()
    dm.setup("fit")
    return dm
end

function pretrain(path::AbstractString; tokenizer="smirk", canonical::Bool=false)
    dm = PY_DATAMODULE[].RobertaDataSet(
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

function __init__()
    ENV["TOKENIZERS_PARALLELISM"] = "false" # Want one CPU per rank
    PY_JSONNET[] = pyimport("_jsonnet")
    PY_DISTRIBUTED[] = pyimport("datasets.distributed")
    PY_DATAMODULE[] = pyimport("electrolyte_fm.data_modules")
    return nothing
end

include("collect.jl")
include("plotting.jl")
include("finetune.jl")

end
