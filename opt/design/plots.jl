using PythonCall

# Load the MIST models
# Normally, I'd do all of the code loading first. By something's funky
# with loading the models later on (seg-faults) so here we are
MISTFinetuned = pyimport("electrolyte_fm.models.prod_finetune").MISTFinetuned
MISTMultiTask = pyimport("electrolyte_fm.models.prod_finetune").MISTMultiTask

models = (
    MISTFinetuned.from_pretrained(joinpath(@__DIR__, "../../models/mist-x4i8qzuq-qm9")),
    "rand" => MISTFinetuned.from_pretrained(joinpath(@__DIR__, "../../models/mist-26.9M-kkgx0omx-qm9")),
    "kt" => MISTFinetuned.from_pretrained(joinpath(@__DIR__, "../../models/mist-26.9M-0vxdbm36-kt/")),
    "kt" => MISTMultiTask.from_pretrained(joinpath(@__DIR__, "../../models/solvent-properties")) => [:pKa],
    MISTFinetuned.from_pretrained(joinpath(@__DIR__, "../../models/mist-26.9M-6hk5coof-dn")),
    MISTFinetuned.from_pretrained(joinpath(@__DIR__, "../../models/mist-26.9M-b302p09x-bp")),
    MISTFinetuned.from_pretrained(joinpath(@__DIR__, "../../models/mist-26.9M-y3ge5pf9-mp")),
    MISTFinetuned.from_pretrained(joinpath(@__DIR__, "../../models/mist-26.9M-cyuo2xb6-fp")),
    "lyte" => MISTMultiTask.from_pretrained(joinpath(@__DIR__, "../../models/electrolyte-solvent/")),
)

# Generate Plots
using Makie
using DesignRules
using DesignRules: simple_hydrocarbons
using MISTStyle
using DataFrames
using CSV: CSV
using StatsBase: mean

# Evaluate hydrocarbons
df_hydrocarbons = DesignRules.predict_all(
    DesignRules.simple_hydrocarbons(30),
    models...;
    n=10
)

for qm_model in ["", "_rand"]
    with_theme(MISTStyle.theme()) do
        DesignRules.hydrocarbon_trends(df_hydrocarbons; qm_model)
    end |> MISTStyle.savefig("hydrocarbons" * qm_model)
end

df_sat = DesignRules.predict_all(
    DesignRules.saturated_fats(24; n_max=9, d_max=6),
    models...;
    n=3,
)

for qm_model in ["", "_rand"]
    for omega in [3, 6, 9]
        with_theme(MISTStyle.theme()) do
            DesignRules.figure_fatty_acids(df_sat; omega, qm_model)
        end |> MISTStyle.savefig("omega-$omega-saturated-fats$qm_model")
    end
end


# Evaluate electrolytes
df_electrolyte = DataFrame(CSV.File("electrolytes.csv"))
df_electrolyte.smi .= DesignRules.encode.(df_electrolyte.smi; encoding="smiles-kekule")
df_electrolyte = DesignRules.predict_all(df_electrolyte, models...; n=10)
df_pc = DesignRules.pubchem_from_jsonl("electrolytes.jsonl")
df_electrolyte = leftjoin(df_electrolyte, df_pc, on=:smi)

with_theme(MISTStyle.theme()) do
    DesignRules.electrolyte_trends(df_electrolyte)
end |> MISTStyle.savefig("electrolytes")

# Save predictions
transform(
    select(df_electrolyte, [:smi, :homo, :gap, :mp, :bp]),
    :homo => ByRow(mean),
    :gap => ByRow(mean),
    :mp => ByRow(mean),
    :bp => ByRow(mean),
) |> CSV.write(joinpath(@__DIR__, "electrolytes_predictons.csv"))

# Permutation sensitivity
df_perm = simple_hydrocarbons(25)
sample_smi(m) = :smi => ByRow(smi -> DesignRules.sample_encodings(smi, m)) => AsTable
df_perm_ref = transform(df_perm, sample_smi(models[1]))          # QM9 finetuned on kekule
df_perm_rand = transform(df_perm, sample_smi(last(models[2])))   # QM9 finetuned with random

# Order sensitivity
df_order_rand = DesignRules.alkene_sweep(25, last(models[2]))
df_order = DesignRules.alkene_sweep(25, models[1])

with_theme(MISTStyle.theme()) do
    DesignRules.figure_permutations(
        "Baseline" => df_perm_ref, "Random" => df_perm_rand;
        name_df_order=("Baseline" => df_order, "Random" => df_order_rand)
    )
end |> MISTStyle.savefig("permutations")


with_theme(MISTStyle.theme()) do
    DesignRules.figure_double_bond_loc(
        "Baseline" => df_perm_ref, "Random" => df_perm_rand;
        name_df_order=("Baseline" => df_order, "Augmented" => df_order_rand)
    )
end |> MISTStyle.savefig("double_bond_loc")
