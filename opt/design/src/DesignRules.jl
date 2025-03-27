module DesignRules

<<<<<<< HEAD
<<<<<<< HEAD
using PythonCall: PythonCall, Py, PyList, pyconvert, @pyconst, pyimport
=======
using PythonCall: PythonCall, Py, PyList, pyconvert, @pyconst
>>>>>>> db6a992 (refactor: Move MistStyle into a proper package)
using DataFrames
using Makie
using MISTStyle
using MISTStyle: label, sublabel!
=======
using PythonCall: PythonCall, Py, PyList, pyconvert, @pyconst, pyimport
using DataFrames
using Makie
using MISTStyle
using MISTStyle: label
>>>>>>> 133d7c3 (feat: better plots for design trends)
using Statistics: Statistics, mean, std
<<<<<<< HEAD
using StatsBase: StatsBase, stderror, mean_and_std, mean, range
using CategoricalArrays: levelcode, categorical
using JSON: JSON

const HARTREE_TO_EV = 27.211_386_245_981
const JOULES_TO_CALORIES = inv(4.184)

sigmoid(x) = 1 / (1 + exp(-x))
=======
using StatsBase: StatsBase, stderror, mean_and_std, mean
using CategoricalArrays: levelcode, categorical

const HARTREE_TO_EV = 27.211_386_245_981
>>>>>>> db6a992 (refactor: Move MistStyle into a proper package)

include("uq.jl")
include("inference.jl")
include("pubchem.jl")

include("hydrocarbons.jl")
include("plot_utils.jl")
include("trends.jl")

end
