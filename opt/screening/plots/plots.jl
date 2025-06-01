using ScreeningPlots
using Makie
using MISTStyle
using StatsBase
using DataFrames
using GLM
using RegressionTables: LatexTable, regtable
using JSON: JSON
using CSV: CSV

ROOTDIR = joinpath(pkgdir(ScreeningPlots), "..")
fig_dir = joinpath(ROOTDIR, "fig")
isdir(fig_dir) || mkdir(fig_dir)

df = ScreeningPlots.collate_performance_stats("initial-sweep")
df.limit_ref_fragments .= something.(df.limit_ref_fragments, 122)
df.limit_db_fragments .= something.(df.limit_db_fragments, 88_800_000)
df.limit_ref_fragments ./= 122
df.limit_db_fragments ./= 88_800_000
df.rel_epoch_size .= df.n_fragments ./ df.epoch_size
transform!(df,
    :trace => ByRow(length) => :trace_samples,
    :rank_throughput => ByRow(mean),
    :rank_throughput => ByRow(sum) => :global_throughput,
    [:n_passing, :unique_molecules] => ByRow(/) => :yield_world,
    [:unique_molecules, :duration] => ByRow(/) => :global_unique_throughput,
)
df.generation_efficiency = df.global_unique_throughput ./ df.global_throughput

# Linear Models to guide scaling
m_eff = lm(
    @formula(generation_efficiency ~ log(gpus) + epoch_size + epoch_size^2 + limit_ref_fragments + limit_db_fragments),
    df
)
display(m_eff)
coef_m = Dict(zip(coefnames(m_eff), coef(m_eff)))
ideal_rel_epoch = -coef_m["epoch_size"] / (2 * coef_m["epoch_size ^ 2"])
@info "Search Metrics" ideal_rel_epoch


m_speed = lm(
    @formula(log(global_throughput) ~ log(gpus) + batch_size + +batch_size^2 + n_fragments + epoch_size),
    df
)
display(m_speed)
coef_speed = Dict(zip(coefnames(m_speed), coef(m_speed)))
ideal_batch_size = -coef_speed["batch_size"] / (2 * coef_speed["batch_size ^ 2"])
@info "Scaling Metrics" ideal_batch_size

regtable(
    m_speed, m_eff;
    file=joinpath(fig_dir, "screening_lm.tex"),
    render=LatexTable(),
)

# Plot generated molecules
production_run = first(sort!(df, :n_passing; rev=true)).path
df_mol = ScreeningPlots.load_generated_molecules(production_run)
prod_config = JSON.parsefile(joinpath(production_run, "config.json"))

# Reference Molecules
df_ref = DataFrame(CSV.File(joinpath(ROOTDIR, "electrolytes.csv")))
df_mol.inchi_key = ScreeningPlots.inchi_key.(df_mol.smiles)
df_ref.inchi_key = ScreeningPlots.inchi_key.(df_ref.smi)

df_novel = subset(df_mol, :inchi_key => ByRow(∉(df_ref.inchi_key)))
df_unfound = subset(df_ref, :inchi_key => ByRow(∉(df_mol.inchi_key)))
@info "Novel Molecules" nrow(df_novel) nrow(df_ref) nrow(df_novel) / nrow(df_mol) nrow(df_unfound) / nrow(df_ref)

