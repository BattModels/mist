using PythonCall

# Load the MIST models
# Normally, I'd do all of the code loading first. By something's funky
# with loading the models later on (seg-faults) so here we are
MISTFinetuned = pyimport("electrolyte_fm.models.prod_finetune").MISTFinetuned
MISTMultiTask = pyimport("electrolyte_fm.models.prod_finetune").MISTMultiTask

mist_qm9 = MISTFinetuned.from_pretrained("../../models/mist-x4i8qzuq-qm9")
mist_kt = MISTMultiTask.from_pretrained("../../models/solvent-properties")
mist_dn = MISTMultiTask.from_pretrained("../../models/donor-number")
mist_solvent = MISTMultiTask.from_pretrained("../../models/mist-solventnet")

# Generate Plots
using Makie
using DesignRules
using DataFrames
using CSV: CSV

include("../style.jl")


# Evaluate hydrocarbons
df_hydrocarbons = DesignRules.predict_all(
    DesignRules.simple_hydrocarbons(25),
    mist_qm9,
    "kt" => mist_kt,
    mist_dn,
    mist_solvent => [:bp, :mp, :fp];
    n=10
)

DesignRules.hydrocarbon_trends(df_hydrocarbons)

# Evaluate electrolytes
df_electrolyte = DataFrame(CSV.File("electrolytes.csv"))
df_electrolyte = DesignRules.predict_all(
    df_electrolyte,
    mist_qm9,
    "kt" => mist_kt,
    mist_dn,
    mist_solvent => [:bp, :mp, :fp];
    n=10
)






