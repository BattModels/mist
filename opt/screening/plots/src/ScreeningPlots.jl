module ScreeningPlots

using CairoMakie
using DataFrames
using Random: randperm
using Graphs: complete_graph, boruvka_mst
using Metaheuristics: Metaheuristics
using JSON: JSON
using SQLite: SQLite
using CSV: CSV
using PythonCall: Py, PyList, pyimport, pyconvert
using GLM: @formula, lm, glm, Normal, LogLink, coef
using Format: format
using Statistics: cor, mean, std
using CategoricalArrays: categorical, levelcode
using LinearAlgebra: norm, dot
using ManifoldLearning: ManifoldLearning, DiffMap, fit, predict, transform
using PrettyTables: LatexTableFormat, LatexCell, pretty_table
using Clustering: hclust

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

mae(x, y) = mean(x -> abs(-(x...)), zip(x, y))
mae(x) = mean(abs, x)
rmsd(x, y) = sqrt(mean(x -> -(x...)^2, zip(x, y)))

default_device() = Sys.isbsd() ? "mps" : "cuda"
function load_mist_pretrained(folder::String; device=default_device())
    return __prod_finetune[].MISTFinetuned.from_pretrained(folder).to(device).eval()
end
function load_mol_surprise(path::String; device=default_device())
    return __mol_surprise[].MolSurpriseFM.from_pretrained(path).to(device).eval()
end

include("qmist.jl")
include("creativity.jl")
include("collate.jl")
include("sqlite.jl")
include("pareto.jl")
include("odor.jl")
include("xyz.jl")
include("lithium_air.jl")

function canonicalize(smi::String)
    mol = __rdkit_chem[].MolFromSmiles(smi)
    isnothing(mol) && return smi
    return pyconvert(String, __rdkit_chem[].MolToSmiles(mol))
end

function isomeric_smiles(smi::String)
    mol = __rdkit_chem[].MolFromSmiles(smi; sanitize=false)
    isnothing(mol) && return smi
    for atom in mol.GetAtoms()
        atom.SetChiralTag(__rdkit_chem[].ChiralType.CHI_UNSPECIFIED)
    end

    return pyconvert(String, __rdkit_chem[].MolToSmiles(mol, isomericSmiles=true))
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
