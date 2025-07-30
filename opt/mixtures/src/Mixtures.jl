module Mixtures

using Makie
using LinearAlgebra: tril!
using MISTStyle: MISTStyle, pt
using DataFrames
using PythonCall: PythonCall, Py, pyconvert, pyimport

pyexcess = Ref{Py}()

function __init__()
    pyexcess[] = pyimport("excess")
    return nothing
end

include("python.jl")
include("skew.jl")
include("dataset.jl")

end
