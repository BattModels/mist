module TokenizerStats

using DataFrames
using Makie
using PythonCall: Py, pyimport, pyconvert, @pyconst
using ArgParse: ArgParseSettings, parse_args, @add_arg_table!
using OnlineStats: OnlineStats, CountMap, HyperLogLog, Extrema, KHist, Counter, fit!, merge!, value
using LinearAlgebra: normalize
using StatsBase: StatsBase, Histogram, fit, AbstractWeights, Weights, nobs, mean, std
using MPI: MPI
using BSON: BSON
using JSON: JSON
using SHA: SHA
using SparseArrays: sparse
using LogExpFunctions: logsumexp, log1pexp, xexpy
using Serialization: serialize, deserialize
using FreeTypeAbstraction: FreeTypeAbstraction, newface, FTFont

function find(dir, pattern)
    found = String[]
    for (root, _, files) in walkdir(dir)
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

function molnet(name::AbstractString; tokenizer="smirk", encoding::String="smiles")
    data_modules = @pyconst(pyimport("electrolyte_fm.data_modules"))
    target_columns = String[]
    dm = data_modules.MolNetDataModule(name; tokenizer, encoding, target_columns, include_encoding=true)
    dm.prepare_data()
    dm.setup("fit")
    return dm
end

function tmqm(path::AbstractString; tokenizer="smirk", encoding::String="smiles")
    data_modules = @pyconst(pyimport("electrolyte_fm.data_modules"))
    dm = data_modules.tmQMDataModule(path, tokenizer; encoding, include_encoding=true)
    dm.prepare_data()
    dm.setup("fit")
    return dm
end

function pretrain(path::AbstractString; tokenizer="smirk", encoding::String="smiles")
    data_modules = @pyconst(pyimport("electrolyte_fm.data_modules"))
    dm = data_modules.RobertaDataSet(path, tokenizer; encoding)
    dm.prepare_data()
    dm.setup("fit")
    return dm
end

include("ngrams.jl")
include("collect.jl")
include("finetune.jl")
include("cli.jl")

include("analysis.jl")
include("plotting.jl")

end
