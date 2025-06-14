module ScreeningPlots

using Makie
using DataFrames
using Metaheuristics: Metaheuristics
using JSON: JSON
using SQLite: SQLite
using PythonCall: Py, pyimport, pyconvert
using GLM: @formula, lm, glm, Normal, LogLink, coef
using Format: format
using StatsBase: cor, mad

using MISTStyle

const HARTREE_TO_EV = 27.211_386_245_981

const __rdkit_chem = Ref{Py}()

function __init__()
    __rdkit_chem[] = pyimport("rdkit.Chem")
end

include("qmist.jl")
include("collate.jl")
include("sqlite.jl")
include("pareto.jl")

function canonicalize(smi::String)
    mol = __rdkit_chem[].MolFromSmiles(smi)
    isnothing(mol) && return smi
    return pyconvert(String, __rdkit_chem[].MolToSmiles(mol))
end

function inchi_key(smi::String)
    mol = __rdkit_chem[].MolFromSmiles(smi)
    @assert !isnothing(mol) lazy"Failed to parse SMILES: $smi"
    return pyconvert(String, __rdkit_chem[].MolToInchiKey(mol))
end

end
