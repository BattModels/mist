module DesignRules

using PythonCall: PythonCall, Py, PyList, pyconvert, @pyconst, pyimport
using DataFrames
using Makie
using MISTStyle
using MISTStyle: label, sublabel!
using Statistics: Statistics, mean, std
using StatsBase: StatsBase, stderror, mean_and_std, mean, range, variation
using CategoricalArrays: levelcode, categorical
using JSON: JSON

const HARTREE_TO_EV = 27.211_386_245_981
const JOULES_TO_CALORIES = inv(4.184)

sigmoid(x) = 1 / (1 + exp(-x))

include("uq.jl")
include("inference.jl")
include("pubchem.jl")

include("hydrocarbons.jl")
include("plot_utils.jl")
include("trends.jl")

end
