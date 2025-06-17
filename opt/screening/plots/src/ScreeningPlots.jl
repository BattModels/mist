module ScreeningPlots

using Makie
using DataFrames
using Metaheuristics: Metaheuristics
using JSON: JSON
using SQLite: SQLite
using PythonCall: Py, PyList, pyimport, pyconvert
using GLM: @formula, lm, glm, Normal, LogLink, coef
using Format: format
using StatsBase: cor, mad

using MISTStyle

const HARTREE_TO_EV = 27.211_386_245_981

const __rdkit_chem = Ref{Py}()
const __mol_surprise = Ref{Py}()
const __prod_finetune = Ref{Py}()
const __data_utils = Ref{Py}()

function __init__()
    __rdkit_chem[] = pyimport("rdkit.Chem")
    __mol_surprise[] = pyimport("electrolyte_fm.models.mol_surprise")
    __prod_finetune[] = pyimport("electrolyte_fm.models.prod_finetune")
    __data_utils[] = pyimport("electrolyte_fm.data_modules.utils")
end

include("qmist.jl")
include("creativity.jl")
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

function searchfirst(f, x)
    idx = findfirst(f, x)
    return isnothing(idx) ? nothing : x[idx]
end

end
