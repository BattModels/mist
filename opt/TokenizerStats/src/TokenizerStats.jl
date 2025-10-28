module TokenizerStats

using Tracy
using PythonCall: Py, pyimport, pyisinstance, pybuiltins, pyconvert, @pyconst
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
const __tokenizer = Ref{Py}()
const pymodel = Ref{Py}()

function __init__()
    __loader[] = pyimport("helper.loader")
    __tokenizer[] = pyimport("helper.tokenizer")
    pymodel[] = pyimport("helper.model")
    return nothing
end

tokenizer_dataset(args...; kwargs...) = __loader[].tokenizer_dataset(args...; kwargs...)
load_tokenizer(name_or_path::AbstractString) = __tokenizer[].load_tokenizer(String(name_or_path))

struct DatasetConfig
    name_or_path::String
    tokenizer::String
    encoding::String
end

function dataset_split(dc::DatasetConfig, split::String; kwargs...)
    split = split == "val" ? "validation" : split
    ds = tokenizer_dataset(dc.tokenizer, dc.name_or_path, dc.encoding; kwargs...)[split]
    return ds
end

tokenizer_name(dc::DatasetConfig) = isdir(dc.tokenizer) ? basename(dc.tokenizer) : dc.tokenizer
function tokenizer(dc::DatasetConfig)
    tok = load_tokenizer(dc.tokenizer)
    info = (;
        tokenizer_name = isdir(dc.tokenizer) ? basename(dc.tokenizer) : dc.tokenizer,
        vocab_size=pyconvert(Int, length(tok)),
        unk_token_id=pyconvert(Union{Int,Nothing}, tok.unk_token_id),
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

include("ngrams.jl")
include("serialize.jl")
include("collect.jl")
include("finetune.jl")

# Attention Maps
include("attention_maps.jl")

include("cli.jl")

end
