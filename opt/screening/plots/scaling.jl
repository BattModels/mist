#!/usr/bin/env -S julia +release --color=auto --startup-file=no --project=@script
using ArgParse
using ScreeningPlots
using Makie
using MISTStyle
using Statistics
using DataFrames
using GLM
using RegressionTables: RegressionTables, LatexTable, regtable
using CSV: CSV

function (@main)(args)
    settings = ArgParseSettings(prog=basename(@__FILE__), description="Collect screening scaling statistics and generate scaling plots.")
    @add_arg_table! settings begin
        "SWEEP_DIR"
            help = "Directory containing scaling runs."
            arg_type = String
            required = true
        "FIG_DIR"
            help = "Output directory for scaling figures and tables."
            arg_type = String
            required = true
        "BP_MODEL"
            help = "Boiling-point model path used in the run configurations."
            arg_type = String
            required = true
        "MP_MODEL"
            help = "Melting-point model path used in the run configurations."
            arg_type = String
            required = true
        "QM9_MODEL"
            help = "QM9 model path used in the run configurations."
            arg_type = String
            required = true
        "FRAGMENT_COUNTS"
            help = "Optional reference and database inventory counts, in that order; omit when limits are fractions."
            arg_type = Int
            nargs = 2
            default = Int[]
            range_tester = count -> count > 0
    end
    parsed = parse_args(args, settings)
    sweep_dir = parsed["SWEEP_DIR"]
    fig_dir = parsed["FIG_DIR"]
    bp_model = parsed["BP_MODEL"]
    mp_model = parsed["MP_MODEL"]
    qm9_model = parsed["QM9_MODEL"]
    counts = parsed["FRAGMENT_COUNTS"]
    reference_fragment_count = isempty(counts) ? nothing : counts[1]
    database_fragment_count = isempty(counts) ? nothing : counts[2]

    mkpath(fig_dir)

    # Filter to screening runs
    df = ScreeningPlots.collate_performance_stats(sweep_dir)
    df = filter(df) do row
        config = row.config
        get(config, "limit_walltime", nothing) == 300 || return false
        models = Set([c["model_path"] for c in config["critics"]])
        expected_models = [bp_model, mp_model, qm9_model]
        models == Set(expected_models) || return false
        return true
    end
    if isnothing(reference_fragment_count)
        df.limit_ref_fragments .= 1.0
        df.limit_db_fragments .= something.(df.limit_db_fragments, 1.0)
    else
        df.limit_ref_fragments .= something.(df.limit_ref_fragments, reference_fragment_count)
        df.limit_db_fragments .= something.(df.limit_db_fragments, database_fragment_count)
        df.limit_ref_fragments ./= reference_fragment_count
        df.limit_db_fragments ./= database_fragment_count
    end
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

    CSV.write(joinpath(fig_dir, "scaling.csv"), select(df, Not([:config, :trace, :rank_throughput])))
    with_theme(MISTStyle.theme()) do
        f = ScreeningPlots.weak_scaling(subset(df, :duration => ByRow(<(400))))
        MISTStyle.savefig("scaling", f; fig_dir)
    end
    return nothing
end
