module TokenizerStats

using Makie
using PythonCall
using OnlineStats: OnlineStats, CountMap, HyperLogLog, Extrema, KHist, fit!, merge!
using LinearAlgebra: normalize
using StatsBase: StatsBase, Histogram, fit, AbstractWeights
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

include("collect.jl")
include("plotting.jl")

end
