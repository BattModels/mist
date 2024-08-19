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
const PY_TOKENIZER = Ref{Py}()
const PY_DISTRIBUTED = Ref{Py}()
const PY_DATASETS = Ref{Py}()

load_tokenizer(name; kwargs...) = PY_TOKENIZER[].load_tokenizer(name; kwargs...)
rdkit_canonical(smi) = PY_TOKENIZER[].rdkit_canonical(smi)
split_dataset_by_node(args...; kwargs...) = PY_DISTRIBUTED[].split_dataset_by_node(args...; kwargs...)
load_dataset(args...; kwargs...) = PY_DATASETS[].load_dataset(args...; kwargs...)

function __init__()
    ENV["TOKENIZERS_PARALLELISM"] = "false" # Want one CPU per rank
    PY_TOKENIZER[] = pyimport("electrolyte_fm.utils.tokenizer")
    PY_DISTRIBUTED[] = pyimport("datasets.distributed")
    PY_DATASETS[] = pyimport("datasets")
    return nothing
end

include("collect.jl")
include("plotting.jl")
include("finetune.jl")

end
