module DesignRules

using PythonCall: PythonCall, Py, PyList, pyconvert, @pyconst, pyimport
using DataFrames
using Makie
using MISTStyle
using MISTStyle: label
using Statistics: Statistics, mean, std
using StatsBase: StatsBase, stderror, mean_and_std, mean
using CategoricalArrays: levelcode, categorical

const HARTREE_TO_EV = 27.211_386_245_981

include("uq.jl")
include("inference.jl")

include("hydrocarbons.jl")
include("plot_utils.jl")
include("trends.jl")

end
