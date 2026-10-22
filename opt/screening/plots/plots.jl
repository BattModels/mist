#!/usr/bin/env -S julia +release --color=auto --startup-file=no --project=@script
using ArgParse
using ScreeningPlots
using Makie
using MISTStyle
using Statistics
using DataFrames
using CSV: CSV
using Format: format

function figure_odor_filter(df_so, df_odor_counts)
    f = Figure(; size=(3inch, 3inch), figure_padding=(3pt, 3pt, 3pt, 3pt))
    ScreeningPlots.plot_pareto_front_scent!(GridLayout(f[1:2, 1]), df_so, "odorless")
    ScreeningPlots.figure_odor_counts!(GridLayout(f[1:2, 2]), df_odor_counts)
    sublabel!(f[1, 1, TopLeft()], "a"; left=5pt)
    sublabel!(f[2, 1, TopLeft()], "b"; left=5pt, down=10pt)
    sublabel!(f[1, 2, TopLeft()], "c"; left=5pt)
    return f
end

function (@main)(args)
    settings = ArgParseSettings(prog=basename(@__FILE__), description="Generate electrolyte screening plots and analysis.")
    @add_arg_table! settings begin
        "RUN_DIR"
            help = "Screening run directory."
            arg_type = String
            required = true
        "REFERENCE_CSV"
            help = "Reference molecules or precomputed reference predictions."
            arg_type = String
            required = true
        "DATA_DIR"
            help = "Released screening and quantum-validation data directory."
            arg_type = String
            required = true
        "FIG_DIR"
            help = "Output directory for figures and tables."
            arg_type = String
            required = true
        "BASE_MODEL"
            help = "Pretrained base-model directory."
            arg_type = String
            required = true
        "BP_MODEL"
            help = "Boiling-point model directory."
            arg_type = String
            required = true
        "MP_MODEL"
            help = "Melting-point model directory."
            arg_type = String
            required = true
        "QM9_MODEL"
            help = "QM9 model directory."
            arg_type = String
            required = true
        "ODOUR_MODEL"
            help = "Odour model directory."
            arg_type = String
            required = true
        "--predict-reference"
            help = "Predict reference properties with the supplied models."
            action = :store_true
        "--quantum-validation"
            help = "Include comparisons with released quantum-validation results."
            action = :store_true
    end
    parsed = parse_args(args, settings)
    production_run_path = parsed["RUN_DIR"]
    reference_csv = parsed["REFERENCE_CSV"]
    DATA_DIR = parsed["DATA_DIR"]
    fig_dir = parsed["FIG_DIR"]
    base_model = parsed["BASE_MODEL"]
    bp_model = parsed["BP_MODEL"]
    mp_model = parsed["MP_MODEL"]
    qm9_model = parsed["QM9_MODEL"]
    odour_model = parsed["ODOUR_MODEL"]
    predict_reference = parsed["predict-reference"]
    quantum_validation = parsed["quantum-validation"]

    mkpath(fig_dir)

    # Plot generated molecules
    prod_id = basename(production_run_path)
    df_mol = ScreeningPlots.load_generated_molecules(production_run_path)

    # Reference Molecules
    df_ref = DataFrame(CSV.File(reference_csv))
    df_ref.smi = String.(df_ref.smi)
    if predict_reference
        for model_path in [bp_model, mp_model, qm9_model]
            model = ScreeningPlots.load_mist_pretrained(model_path)
            predictions = ScreeningPlots.predict_mist(model, String.(df_ref.smi))
            for column in names(predictions, Not(:smiles))
                df_ref[!, column] = predictions[!, column]
            end
        end
        CSV.write(joinpath(fig_dir, "reference_predictions.csv"), df_ref)
    end
    df_mol.inchi_key = ScreeningPlots.inchi_key.(df_mol.smiles)
    df_ref.inchi_key = ScreeningPlots.inchi_key.(df_ref.smi)

    df_novel = subset(df_mol, :inchi_key => ByRow(∉(df_ref.inchi_key)))
    df_unfound = subset(df_ref, :inchi_key => ByRow(∉(df_mol.inchi_key)))
    @info "Novel Molecules" nrow(df_novel) nrow(df_ref) nrow(df_novel) / nrow(df_mol) nrow(df_unfound) / nrow(df_ref)

    # Save Pareto Front
    table, df_front = ScreeningPlots.save_pareto_front(df_mol)
    CSV.write(joinpath(fig_dir, "generated_pareto_front.csv"), df_front)
    write(joinpath(fig_dir, "pareto_front_table.tex"), table)
    open(joinpath(fig_dir, "pareto_front_smiles.txt"), "w") do io
        foreach(smiles -> println(io, smiles), df_front.smiles)
    end

    # Generate Plots
    trace, _ = ScreeningPlots.performance_trace(joinpath(production_run_path, "screen.jsonl"))
    trace = DataFrame(trace)
    with_theme(MISTStyle.theme()) do
        f = ScreeningPlots.plot_pareto_front(df_mol, df_ref)
        MISTStyle.savefig("production-" * prod_id, f; fig_dir)

        f = ScreeningPlots.plot_gen_trace(trace)
        MISTStyle.savefig("gen-trace-" * prod_id, f; fig_dir)

        if quantum_validation
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
                ScreeningPlots.figure_parity(df, cols; label) |> MISTStyle.savefig(basename(version) * "_parity"; fig_dir)
            end

            # Parity Plots vs. QM9 Calculations
            for (dir_name, label) in ["qm9" => "rdkit", "qm9_obabel" => "openbabel", "qm9_conf" => "conformer"]
                f = ScreeningPlots.compare_qmist(
                    production_run_path,
                    joinpath(production_run_path, dir_name);
                    label="B3LYP/6-31G(2df,p) - Ours (v3)" => "MIST",
                )
                MISTStyle.savefig("parity-$(label)-$(prod_id)", f; fig_dir)
            end

            # Load the QM9 Model used for screening
            mist_qm9 = ScreeningPlots.load_mist_pretrained(qm9_model)
            mist_qm9 = mist_qm9.to(ScreeningPlots.default_device())
            #
            # Parity Plots with Chembl data
            f = ScreeningPlots.compare_qmist(
                production_run_path,
                joinpath(production_run_path, "qm9_conf"),
                joinpath(DATA_DIR, "veri_chembl"),
                mist_qm9;
                label="B3LYP/6-31G(2df,p)" => "MIST",
            )
            MISTStyle.savefig("parity-chembl-$(prod_id)", f; fig_dir)

        end

    end

    mol_surprise = ScreeningPlots.load_mol_surprise(base_model)
    mist_mp = ScreeningPlots.load_mist_pretrained(mp_model)
    mist_bp = ScreeningPlots.load_mist_pretrained(bp_model)

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
    end |> MISTStyle.savefig("surprise_vs_utility"; fig_dir)

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

    CSV.write(joinpath(fig_dir, "renyi_entropy.csv"), df_s)

    # Bar chart of top odor in generated molecules
    odor_model = ScreeningPlots.load_mist_pretrained(odour_model)
    df_odor = select(df_surprise, :smiles, :group)
    df_odor = innerjoin(df_odor, ScreeningPlots.predict_mist(odor_model, df_odor.smiles); on=:smiles)
    df_odor_counts = ScreeningPlots.odor_counts(subset(df_odor, :group => ByRow(!=("ChEMBL"))), odor_model)
    df_so = innerjoin(df_surprise, df_odor; on=["smiles", "group"])
    with_theme(MISTStyle.theme()) do
        f = ScreeningPlots.figure_odor_counts(df_odor_counts)
        MISTStyle.savefig("screening_odors", f; fig_dir)

        f = ScreeningPlots.plot_pareto_front_scent(df_so, "odorless")
        MISTStyle.savefig("electrolyte_odorless", f; fig_dir)

        f = figure_odor_filter(df_so, df_odor_counts)
        MISTStyle.savefig("screening_olfaction", f; fig_dir)
    end
    return nothing
end
