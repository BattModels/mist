module Mixtures

using Makie
using LinearAlgebra: tril!
using MISTStyle: MISTStyle, pt
using DataFrames
using PythonCall: PythonCall, Py, pyconvert, pyimport

pyexcess = Ref{Py}()
pyionic = Ref{Py}()


function __init__()
    pyexcess[] = pyimport("excess")
    pyionic[] = pyimport("ionic_conductivity")
    return nothing
end

include("python.jl")
include("skew.jl")
include("dataset.jl")

end
