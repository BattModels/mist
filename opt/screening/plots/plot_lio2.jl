#!/usr/bin/env -S julia --color=auto --startup-file=no --project=@script
using ScreeningPlots
using CairoMakie
using MISTStyle
using PythonCall: pyimport
using CSV: CSV
using DataFrames: subset, ByRow, nrow

ROOTDIR = realpath(joinpath(pkgdir(ScreeningPlots), ".."))
GIT_ROOT = realpath(joinpath(ROOTDIR, "..", ".."))
fig_dir = joinpath(ROOTDIR, "fig")
isdir(fig_dir) || mkdir(fig_dir)

run_path = "runs/screening/out/3e6b639f-6cd8-40a6-947f-62b238ab408d"
df = ScreeningPlots.load_generated_molecules(run_path)

# Rename DMSO_pKa to pKa_DMSO for consistency
if hasproperty(df, :DMSO_pKa)
    df.pKa_DMSO = df.DMSO_pKa
end

# Add screening directory to Python path
screening_dir = joinpath(GIT_ROOT, "opt", "screening")
sys = pyimport("sys")
sys.path.insert(0, screening_dir)

equations = pyimport("lithium_air_equations")
df_ref = ScreeningPlots.load_lithium_air_reference(
    joinpath(screening_dir, "lithium_air_reference_solvents.csv"),
    joinpath(GIT_ROOT, "data", "models"),
    equations
)

df.inchi_key = ScreeningPlots.inchi_key.(df.smiles)
df_ref.inchi_key = ScreeningPlots.inchi_key.(df_ref.smi)

df_novel = subset(df, :inchi_key => ByRow(∉(df_ref.inchi_key)))
df_unfound = subset(df_ref, :inchi_key => ByRow(∉(df.inchi_key)))
@info "Novel Molecules" nrow(df_novel) nrow(df_ref) nrow(df_novel)/nrow(df) nrow(df_unfound)/nrow(df_ref)

table, df_front = ScreeningPlots.save_lithium_air_pareto_front(df)
CSV.write(joinpath(fig_dir, "lio2_pareto_front.csv"), df_front)
# write(joinpath(fig_dir, "lio2_pareto_front_table.tex"), table)

# Write Pareto front SMILES to txt file
open(joinpath(fig_dir, "lio2_pareto_front_smiles.txt"), "w") do io
    for smiles in df_front.smiles
        println(io, smiles)
    end
end

with_theme(MISTStyle.theme()) do
    f = ScreeningPlots.plot_lithium_air_pareto(df, df_ref)
    MISTStyle.savefig(joinpath(fig_dir, "lio2_pareto"), f)

    f_homo = ScreeningPlots.plot_lithium_air_homo_colored(df, df_ref)
    MISTStyle.savefig(joinpath(fig_dir, "lio2_homo_colored"), f_homo)

    f_homo_pka = ScreeningPlots.plot_lithium_air_homo_pka(df, df_ref)
    MISTStyle.savefig(joinpath(fig_dir, "lio2_homo_pka"), f_homo_pka)

    f_pka_habstraction = ScreeningPlots.plot_lithium_air_pka_habstraction(df, df_ref)
    MISTStyle.savefig(joinpath(fig_dir, "lio2_pka_habstraction"), f_pka_habstraction)

    f_nucleophilic = ScreeningPlots.plot_lithium_air_nucleophilic_attack(df, df_ref, equations)
    MISTStyle.savefig(joinpath(fig_dir, "lio2_nucleophilic_attack"), f_nucleophilic)

    f_fgroups = ScreeningPlots.plot_lithium_air_functional_groups(df, df_ref)
    MISTStyle.savefig(joinpath(fig_dir, "lio2_functional_groups"), f_fgroups)
end

println("Plots saved to $fig_dir")
