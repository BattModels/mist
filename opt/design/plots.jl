using PythonCall

# Load the MIST models
# Normally, I'd do all of the code loading first. By something's funky
# with loading the models later on (seg-faults) so here we are
MISTFinetuned = pyimport("electrolyte_fm.models.prod_finetune").MISTFinetuned
MISTMultiTask = pyimport("electrolyte_fm.models.prod_finetune").MISTMultiTask

models = (
    MISTFinetuned.from_pretrained("../../models/mist-x4i8qzuq-qm9"),
    "rand" => MISTFinetuned.from_pretrained("../../models/mist-26.9M-kkgx0omx-qm9"),
    "kt" => MISTMultiTask.from_pretrained("../../models/solvent-properties"),
    MISTFinetuned.from_pretrained("../../models/mist-26.9M-6hk5coof-dn"),
    MISTFinetuned.from_pretrained("../../models/mist-26.9M-b302p09x-bp"),
    MISTFinetuned.from_pretrained("../../models/mist-26.9M-y3ge5pf9-mp"),
    MISTFinetuned.from_pretrained("../../models/mist-26.9M-cyuo2xb6-fp"),
    "lyte" => MISTMultiTask.from_pretrained("../../models/electrolyte-solvent/"),
)

# Generate Plots
using Makie
using DesignRules
using DesignRules: simple_hydrocarbons
using MISTStyle
using DataFrames
using CSV: CSV

# Evaluate hydrocarbons
df_hydrocarbons = DesignRules.predict_all(
    DesignRules.simple_hydrocarbons(25),
    models...;
    n=2
)

with_theme(MISTStyle.theme()) do
    DesignRules.hydrocarbon_trends(df_hydrocarbons)
end

# Evaluate electrolytes
df_electrolyte = DataFrame(CSV.File("electrolytes.csv"))
df_electrolyte.smi .= DesignRules.encode.(df_electrolyte.smi; encoding="smiles-kekule")
df_electrolyte = DesignRules.predict_all(df_electrolyte, models...; n=10)

with_theme(MISTStyle.theme()) do
    DesignRules.electrolyte_trends(df_electrolyte)
end

# Permutation sensitivity
df_perm = simple_hydrocarbons(25)
f(m) = :smi => ByRow(smi -> DesignRules.sample_encodings(smi, m)) => AsTable
df_perm_ref = transform(df_perm, f(models[1]))          # QM9 finetuned on kekule
df_perm_rand = transform(df_perm, f(last(models[2])))   # QM9 finetuned with random

# Order sensitivity
df_order_rand = DesignRules.alkene_sweep(25, last(models[2]))
df_order = DesignRules.alkene_sweep(25, models[1])

with_theme(MISTStyle.theme()) do
    DesignRules.figure_permutations(
        "Kekule" => df_perm_ref, "Random" => df_perm_rand;
        name_df_order=("Kekule" => df_order, "Random" => df_order_rand)
    )
end
