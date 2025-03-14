module DesignRules

using PythonCall: PythonCall, Py, PyList, pyconvert
using DataFrames
using Makie
using Statistics: Statistics, mean, std
using StatsBase: StatsBase, stderror, mean_and_std

export labels, cb_attrs

include("uq.jl")
include("inference.jl")

include("hydrocarbons.jl")
include("plot_utils.jl")

# Helper for getting the label from plot elements
labels(x) = x.label[]

end
