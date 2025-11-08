#!/usr/bin/env -S julia +release --color=auto --startup-file=no --project=@script
# Script to generate screening plots and analysis
using ScreeningPlots
using Makie
using MISTStyle
using Statistics
using DataFrames
using GLM
using RegressionTables: RegressionTables, LatexTable, regtable
using JSON: JSON
using CSV: CSV
using Format: format

using ScreeningPlots: searchfirst

# Input Files and Path
# ROOTDIR: opt/screening folder
# GIT_ROOT: MIST Project Root
# DATA_DIR: Path to the screening files (to be released in data drop)
# MODEL_DIR: Path to the model checkpoints (to be released in data drop)
ROOTDIR = realpath(joinpath(pkgdir(ScreeningPlots), ".."))
GIT_ROOT = realpath(joinpath(ROOTDIR, "..", ".."))
DATA_DIR = joinpath(GIT_ROOT, "data", "screening")
MODEL_DIR = joinpath(GIT_ROOT, "data", "models")
fig_dir = joinpath(ROOTDIR, "fig")
isdir(fig_dir) || mkdir(fig_dir)

# Filter to screening runs
df = ScreeningPlots.collate_performance_stats(joinpath(DATA_DIR, "runs"))
df = filter(df) do row
    config = row.config
    get(config, "limit_walltime", nothing) == 300 || return false
    models = Set([c["model_path"] for c in config["critics"]])
    expected_models = [
        "models/mist-26.9M-b302p09x-bp",
        "models/mist-26.9M-y3ge5pf9-mp",
        "models/mist-x4i8qzuq-qm9",
    ]
    models == Set(expected_models) || return false
    return true
end
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
    subset(df, :duration => ByRow(<(400))),
)
display(m_eff)
coef_m = Dict(zip(coefnames(m_eff), coef(m_eff)))
ideal_rel_epoch = -coef_m["epoch_size"] / (2 * coef_m["epoch_size ^ 2"])
@info "Search Metrics" ideal_rel_epoch


m_speed = lm(
    @formula(log(global_throughput) ~ log(gpus) + batch_size + +batch_size^2 + n_fragments + epoch_size),
    subset(df, :duration => ByRow(<(400))),
)
display(m_speed)
coef_speed = Dict(zip(coefnames(m_speed), coef(m_speed)))
ideal_batch_size = -coef_speed["batch_size"] / (2 * coef_speed["batch_size ^ 2"])
@info "Scaling Metrics" ideal_batch_size

regtable(
    m_speed, m_eff;
    file=joinpath(fig_dir, "screening_lm.tex"),
    render=LatexTable(),
    regression_statistics=[
            RegressionTables.Nobs,
            RegressionTables.DOF,
            RegressionTables.R2,
            (m -> ScreeningPlots.mae(residuals(m))) => "MAE",
            (m -> ScreeningPlots.rmsd(predict(m), response(m))) => "RMSE",
    ]
)

# Plot generated molecules
prod_id = "b7c6ceb2-2114-4ba5-bd2e-41b9bfa2d5df"
production_run_path = joinpath(DATA_DIR, "runs", prod_id)
df_mol = ScreeningPlots.load_generated_molecules(production_run_path)
prod_config = JSON.parsefile(joinpath(production_run_path, "config.json"))

# Reference Molecules
df_ref = DataFrame(CSV.File(joinpath(DATA_DIR, "electrolytes_predictons.csv")))
df_mol.inchi_key = ScreeningPlots.inchi_key.(df_mol.smiles)
df_ref.inchi_key = ScreeningPlots.inchi_key.(df_ref.smi)

df_novel = subset(df_mol, :inchi_key => ByRow(∉(df_ref.inchi_key)))
df_unfound = subset(df_ref, :inchi_key => ByRow(∉(df_mol.inchi_key)))
@info "Novel Molecules" nrow(df_novel) nrow(df_ref) nrow(df_novel) / nrow(df_mol) nrow(df_unfound) / nrow(df_ref)

# Save Pareto Front
table, df_front = ScreeningPlots.save_pareto_front(df_mol)
CSV.write(joinpath(DATA_DIR, "generated_pareto_front.csv"), df_front)
write(joinpath("fig", "pareto_front_table.tex"), table)

# Generate Plots
trace, _ = ScreeningPlots.performance_trace(joinpath(production_run_path, "screen.jsonl"))
trace = DataFrame(trace)
with_theme(MISTStyle.theme()) do
    f = ScreeningPlots.plot_pareto_front(df_mol, df_ref)
    MISTStyle.savefig(joinpath("production" * "-" * prod_id), f)

    f = ScreeningPlots.plot_gen_trace(trace)
    MISTStyle.savefig(joinpath("gen-trace" * "-" * prod_id), f)

    f = ScreeningPlots.weak_scaling(subset(df, :duration => ByRow(<(400))))
    MISTStyle.savefig(joinpath("scaling" * "-" * prod_id), f)

    f = ScreeningPlots.figure_screening(trace, df_mol, df_ref, df)
    MISTStyle.savefig(joinpath("panel" * "-" * prod_id), f)

    # Verify qmist can reproduce QM9 calculations
    df_qm9 = ScreeningPlots.load_jsonl(joinpath(DATA_DIR, "qm9.jsonl"))
    for (version, vlabel) in [
        (joinpath(DATA_DIR, "veri_v1"), "v1"),
        (joinpath(DATA_DIR, "veri_v2"), "v2"),
        (joinpath(DATA_DIR, "veri_v3"), "v3"),
    ]
        df_qmist = ScreeningPlots.load_qmist_results(version)
        df, cols = ScreeningPlots.merge_qmist_results(df_qmist, df_qm9)
        μ = mean(df_qmist.walltime)
        σ = std(df_qmist.walltime)
        walltime_p95 = quantile(df_qmist.walltime, 0.95)
        @info basename(version) nrow(df) walltime=format("\\({:.0f} \\pm {:.0f}\\)", μ, σ) walltime_p95
        label = "QM9 (Ramakrishnan et al.)" => "Ours ($vlabel)"
        ScreeningPlots.figure_parity(df, cols; label) |> MISTStyle.savefig(basename(version) * "_parity")
    end

    # Parity Plots vs. QM9 Calculations
    for (dir_name, label) in ["qm9" => "rdkit", "qm9_obabel" => "openbabel", "qm9_conf" => "conformer"]
        f = ScreeningPlots.compare_qmist(
            production_run_path,
            joinpath(production_run_path, dir_name);
            label="B3LYP/6-31G(2df,p) - Ours (v3)" => "MIST",
        )
        MISTStyle.savefig(joinpath("parity-$(label)-$(prod_id)"), f)
    end

    # Load the QM9 Model used for screening
    mist_qm9 = ScreeningPlots.load_mist_pretrained(joinpath(MODEL_DIR, "mist-ti624ev1-moleculenet", "qm9"))
    mist_qm9 = mist_qm9.to("mps")
    #
    # Parity Plots with Chembl data
    f = ScreeningPlots.compare_qmist(
        production_run_path,
        joinpath(production_run_path, "qm9_conf"),
        joinpath(DATA_DIR, "veri_chembl"),
        mist_qm9;
        label="B3LYP/6-31G(2df,p)" => "MIST",
    )
    MISTStyle.savefig(joinpath("parity-chembl-$(prod_id)"), f)

end

mol_surprise = ScreeningPlots.load_mol_surprise(joinpath(MODEL_DIR, "mist-ti624ev1-moleculenet", "pretrained"))
mist_mp = ScreeningPlots.load_mist_pretrained(joinpath(MODEL_DIR, "mist-26.9M-y3ge5pf9-mp"))
mist_bp = ScreeningPlots.load_mist_pretrained(joinpath(MODEL_DIR, "mist-26.9M-b302p09x-bp"))

_, df_surprise = ScreeningPlots.compare_creativity(
    production_run_path,
    joinpath(DATA_DIR, "veri_chembl"),
    df_ref;
    mol_surprise,
    mist_mp,
    mist_bp
)
with_theme(MISTStyle.theme()) do
    ScreeningPlots.compare_creativity(df_surprise)
end |> MISTStyle.savefig("surprise_vs_utility")

# Compute Renyi Entropy Metrics
df_surprise.embed_mean = eachrow(ScreeningPlots.mist_embedding(mol_surprise, df_surprise.smiles; pooling=ScreeningPlots.mean_pooling))
dist_metrics = [
    "eculidean" => ScreeningPlots.eculidean_distance,
    "cosine" => ScreeningPlots.cosine_distance,
    "angular" => ScreeningPlots.angular_distance,
]
df_s = combine(groupby(df_surprise, :group)) do gdf
    out = []
    n = nrow(gdf)
    for (pool, emb) in ["first" => gdf.embed, "mean" => gdf.embed_mean]
        for (metric, distance) in dist_metrics
            o = ScreeningPlots.renyi_entropy_estimate(emb; distance)
            push!(out, (; n, pool, metric, o...))
        end
    end
    return DataFrame(out)
end

# Bar chart of top odor in generated molecules
odor_model = ScreeningPlots.load_mist_pretrained(joinpath(MODEL_DIR, "mist-26.9M-48kpooqf-odour"))
df_odor = select(df_surprise, :smiles, :group)
df_odor = innerjoin(df_odor, ScreeningPlots.predict_mist(odor_model, df_odor.smiles); on=:smiles)
df_odor_counts = ScreeningPlots.odor_counts(subset(df_odor, :group => ByRow(!=("ChEMBL"))), odor_model)
f = ScreeningPlots.figure_odor_counts(df_odor_counts)
MISTStyle.savefig("screening_odors", f)

# Screen for odorless
df_so = innerjoin(df_surprise, df_odor; on=["smiles", "group"])
f = ScreeningPlots.plot_pareto_front_scent(df_so, "odorless")
MISTStyle.savefig("electrolyte_odorless", f)

function figure_odor_filter(df_so, df_odor_counts)
    f = Figure(; size=(3inch, 3inch), figure_padding=(3pt, 3pt, 3pt, 3pt))
    ScreeningPlots.plot_pareto_front_scent!(GridLayout(f[1:2, 1]), df_so, "odorless")
    ScreeningPlots.figure_odor_counts!(GridLayout(f[1:2, 2]), df_odor_counts)
    sublabel!(f[1, 1, TopLeft()], "a"; left=5pt)
    sublabel!(f[2, 1, TopLeft()], "b"; left=5pt, down=10pt)
    sublabel!(f[1, 2, TopLeft()], "c"; left=5pt)
    return f
end
with_theme(MISTStyle.theme()) do
    figure_odor_filter(df_so, df_odor_counts)
end |> MISTStyle.savefig("screening_olfaction")
