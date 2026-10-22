#!/usr/bin/env -S julia --color=auto --startup-file=no --project=@script
using ArgParse
using ScreeningPlots
using CairoMakie
using MISTStyle
using PythonCall: pyimport
using CSV: CSV
using DataFrames: subset, ByRow, nrow

function (@main)(args)
    settings = ArgParseSettings(prog=basename(@__FILE__), description="Generate lithium-air screening plots and analysis.")
    @add_arg_table! settings begin
        "RUN_DIR"
            help = "Lithium-air screening run directory."
            arg_type = String
            required = true
        "REFERENCE_CSV"
            help = "Reference-solvent CSV."
            arg_type = String
            required = true
        "FIG_DIR"
            help = "Output directory for figures and tables."
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
        "PKA_MODEL"
            help = "pKa model directory."
            arg_type = String
            required = true
        "ETN_MODEL"
            help = "ETN model directory."
            arg_type = String
            required = true
        "KT_MODEL"
            help = "Kamlet-Taft model directory."
            arg_type = String
            required = true
    end
    parsed = parse_args(args, settings)
    run_path = parsed["RUN_DIR"]
    reference_csv = parsed["REFERENCE_CSV"]
    fig_dir = parsed["FIG_DIR"]
    bp_model = parsed["BP_MODEL"]
    mp_model = parsed["MP_MODEL"]
    qm9_model = parsed["QM9_MODEL"]
    pka_model = parsed["PKA_MODEL"]
    etn_model = parsed["ETN_MODEL"]
    kt_model = parsed["KT_MODEL"]

    ROOTDIR = realpath(joinpath(pkgdir(ScreeningPlots), ".."))
    GIT_ROOT = realpath(joinpath(ROOTDIR, "..", ".."))
    mkpath(fig_dir)

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
        reference_csv,
        equations;
        bp_model, mp_model, qm9_model, pka_model, etn_model, kt_model,
        cache_path=joinpath(fig_dir, "reference_predictions.csv"),
    )

    df.inchi_key = ScreeningPlots.inchi_key.(df.smiles)
    df_ref.inchi_key = ScreeningPlots.inchi_key.(df_ref.smi)

    df_novel = subset(df, :inchi_key => ByRow(∉(df_ref.inchi_key)))
    df_unfound = subset(df_ref, :inchi_key => ByRow(∉(df.inchi_key)))
    @info "Novel Molecules" nrow(df_novel) nrow(df_ref) nrow(df_novel)/nrow(df) nrow(df_unfound)/nrow(df_ref)

    table, df_front = ScreeningPlots.save_lithium_air_pareto_front(df)
    CSV.write(joinpath(fig_dir, "lio2_pareto_front.csv"), df_front)
    write(joinpath(fig_dir, "lio2_pareto_front_table.tex"), table)

    # Write Pareto front SMILES to txt file
    open(joinpath(fig_dir, "lio2_pareto_front_smiles.txt"), "w") do io
        for smiles in df_front.smiles
            println(io, smiles)
        end
    end

    with_theme(MISTStyle.theme()) do
        f = ScreeningPlots.plot_lithium_air_pareto(df, df_ref)
        MISTStyle.savefig("lio2_pareto", f; fig_dir)

        f_homo = ScreeningPlots.plot_lithium_air_homo_colored(df, df_ref)
        MISTStyle.savefig("lio2_homo_colored", f_homo; fig_dir)

        f_homo_pka = ScreeningPlots.plot_lithium_air_homo_pka(df, df_ref)
        MISTStyle.savefig("lio2_homo_pka", f_homo_pka; fig_dir)

        f_pka_habstraction = ScreeningPlots.plot_lithium_air_pka_habstraction(df, df_ref)
        MISTStyle.savefig("lio2_pka_habstraction", f_pka_habstraction; fig_dir)

        f_nucleophilic = ScreeningPlots.plot_lithium_air_nucleophilic_attack(df, df_ref, equations)
        MISTStyle.savefig("lio2_nucleophilic_attack", f_nucleophilic; fig_dir)

        f_fgroups = ScreeningPlots.plot_lithium_air_functional_groups(df, df_ref)
        MISTStyle.savefig("lio2_functional_groups", f_fgroups; fig_dir)
    end

    println("Plots saved to $fig_dir")
    return nothing
end
