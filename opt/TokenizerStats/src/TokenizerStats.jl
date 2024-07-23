module TokenizerStats

using Makie
using PythonCall
using OnlineStats
using LinearAlgebra: normalize
using StatsBase: StatsBase, Histogram, fit, AbstractWeights
using MPI: MPI
using JSON: JSON

function find(dir, pattern)
    files = String[]
    for file in readdir(dir; join=true)
        if match(pattern, file) !== nothing
            push!(files, file)
        end
    end
    return files
end

include("collect.jl")
include("plotting.jl")

end
