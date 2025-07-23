module Mixtures

using DataFrames
using PythonCall: PythonCall, Py, pyconvert, pyimport

pyexcess = Ref{Py}()

function __init__()
    pyexcess[] = pyimport("excess")
    return nothing
end

include("python.jl")

end
