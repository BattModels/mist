module TokenizerStats

using PythonCall: Py, pyimport, pyconvert, @pyconst
using ArgParse: ArgParseSettings, parse_args, @add_arg_table!
using OnlineStats: OnlineStats, CountMap, HyperLogLog, Extrema, KHist, Counter, fit!, merge!, value
using LinearAlgebra: normalize
using StatsBase: StatsBase, Histogram, fit, nobs, mean, std
using MPI: MPI
using JLD2: JLD2, jldopen
using JSON: JSON
using SHA: SHA
using Dates: now
using SparseArrays: sparse
using LogExpFunctions: logsumexp, log1pexp, xexpy
using Serialization: serialize, deserialize
using NVTX: @annotate

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


const __loader = Ref{Py}()
function __init__()
    __loader[] = pyimport("helper.loader")
    __tokenizer[] = pyimport("helper.tokenizer")
    return nothing
end

tokenizer_dataset(args...; kwargs...) = __loader[].tokenizer_dataset(args...; kwargs...)
load_tokenizer(args...; kwargs...) = __tokenizer[].load_tokenizer(args...; kwargs...)

struct DatasetConfig
    name_or_path::String
    tokenizer::String
    encoding::String
end

function dataset_split(dc::DatasetConfig, split::String; kwargs...)
    return tokenizer_dataset(dc.tokenizer, dc.name_or_path, dc.encoding; kwargs...)[split]
end

tokenizer_name(dc::DatasetConfig) = isdir(dc.tokenizer) ? basename(dc.tokenizer) : dc.tokenizer
function tokenizer(dc::DatasetConfig)
    tok = load_tokenizer(dc.tokenizer)
    info = (;
        tokenizer_name = isdir(dc.tokenizer) ? basename(dc.tokenizer) : dc.tokenizer,
            vocab_size=pyconvert(Int, length(tokenizer)),
            unk_token_id=pyconvert(Union{Int,Nothing}, tokenizer.unk_token_id),
    )
    return tok, info
end

function dataset_name(dc::DatasetConfig)
    if isdir(dc.name_or_path)
        if "tmQM" in splitpath(dc.name_or_path)
            return "tmQM"
        else
            return basename(dc.name_or_path)
        end
    end
    return dc.name_or_path
end

function split_dataset_by_node(args...; kwargs...)
    m = @pyconst(pyimport("datasets.distributed"))
    return m.split_dataset_by_node(args...; kwargs...)
end

function molnet(name::AbstractString; tokenizer="smirk", encoding::String="smiles")
    data_modules = @pyconst(pyimport("electrolyte_fm.data_modules"))
    target_columns = String[]
    dm = data_modules.MolNetDataModule(name; tokenizer, encoding, target_columns, include_encoding=true)
    dm.prepare_data()
    return dm
end

function tmqm(path::AbstractString; tokenizer="smirk", encoding::String="smiles")
    data_modules = @pyconst(pyimport("electrolyte_fm.data_modules"))
    dm = data_modules.tmQMDataModule(path; tokenizer, encoding, include_encoding=true)
    dm.prepare_data()
    return dm
end

function pretrain(path::AbstractString; tokenizer="smirk", encoding::String="smiles")
    data_modules = @pyconst(pyimport("electrolyte_fm.data_modules"))
    dm = data_modules.RobertaDataSet(path; tokenizer, encoding)
    dm.prepare_data()
    return dm
end

include("ngrams.jl")
include("serialize.jl")
include("collect.jl")
include("finetune.jl")
include("cli.jl")

end
