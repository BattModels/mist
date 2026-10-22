#!/usr/bin/env -S julia +release --color=auto --startup-file=no --project=@script
using ArgParse
using ScreeningPlots
using Makie
using MISTStyle
using DataFrames
using CSV: CSV

function (@main)(args)
    settings = ArgParseSettings(prog=basename(@__FILE__), description="Combine production screening results and scaling statistics in a panel.")
    @add_arg_table! settings begin
        "RUN_DIR"
            help = "Production screening run directory."
            arg_type = String
            required = true
        "REFERENCE_CSV"
            help = "Reference molecule predictions."
            arg_type = String
            required = true
        "SCALING_CSV"
            help = "Scaling statistics produced by scaling.jl."
            arg_type = String
            required = true
        "FIG_DIR"
            help = "Output directory for the combined panel."
            arg_type = String
            required = true
    end
    parsed = parse_args(args, settings)
    run_path = parsed["RUN_DIR"]
    reference_csv = parsed["REFERENCE_CSV"]
    scaling_csv = parsed["SCALING_CSV"]
    fig_dir = parsed["FIG_DIR"]

    mkpath(fig_dir)
    df_mol = ScreeningPlots.load_generated_molecules(run_path)
    df_ref = DataFrame(CSV.File(reference_csv))
    df_speed = DataFrame(CSV.File(scaling_csv))
    df_mol.inchi_key = ScreeningPlots.inchi_key.(df_mol.smiles)
    df_ref.inchi_key = ScreeningPlots.inchi_key.(df_ref.smi)
    trace, _ = ScreeningPlots.performance_trace(joinpath(run_path, "screen.jsonl"))
    with_theme(MISTStyle.theme()) do
        f = ScreeningPlots.figure_screening(DataFrame(trace), df_mol, df_ref, df_speed)
        MISTStyle.savefig("panel-" * basename(run_path), f; fig_dir)
    end
    return nothing
end
