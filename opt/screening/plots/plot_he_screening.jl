#!/usr/bin/env -S julia --color=auto --startup-file=no --project=@.
using ScreeningPlots
using Makie, CairoMakie
using MISTStyle
using DataFrames
using Statistics
using JSON

ROOTDIR = @__DIR__
OUT_DIR = joinpath(ROOTDIR, "..", "out")
FIG_DIR = joinpath(ROOTDIR, "..", "fig")
isdir(FIG_DIR) || mkdir(FIG_DIR)

runs = readdir(OUT_DIR; join=false)

function detonation_velocity_vs_time(df_mol)
    explosive_props = [
        (compound="TNT",   detonation_velocity=7188.5708386335755, logImpact_J=1.6, year=1863),
        # (compound="NG",    detonation_velocity=9005.141655,        logImpact_J=0.69019608, year=1846),
        # (compound="RDX",   detonation_velocity=8954.21062212595,   logImpact_J=0.801197225, year=1899),
        # (compound="HMX",   detonation_velocity=9384.431104421,     logImpact_J=0.82849878, year=1941),
        (compound="TATB",  detonation_velocity=7771.0264871952,    logImpact_J=1.98683911, year=1888),
        (compound="DFP",   detonation_velocity=8834.19714,         logImpact_J=1.25527251, year=2000),
        # (compound="CL-20", detonation_velocity=9946.886017653309,  logImpact_J=0.392954635, year=1987),
    ]

    danger_props = [
        (compound="NG",    detonation_velocity=9005.141655,        logImpact_J=0.69019608, year=1846),
        (compound="RDX",   detonation_velocity=8954.21062212595,   logImpact_J=0.801197225, year=1899),
        (compound="HMX",   detonation_velocity=9384.431104421,     logImpact_J=0.82849878, year=1941),
        (compound="CL-20", detonation_velocity=9946.886017653309,  logImpact_J=0.392954635, year=1987),
    ]

    f = Figure(size=(320, 160), figure_padding=(2, 10, 2, 15))
    ax = Axis(f[1, 1];
        xlabel="Year",
        ylabel="Detonation Velocity [m/s]",
        limits=((1800, 2100), (7000, 10000))
    )

    # Add screened molecules at year 2026
    if nrow(df_mol) > 0
        scatter!(ax,
            fill(2026, nrow(df_mol)),
            df_mol[!, "det_velocity"];
            color=(:blue),
            marker=:star5,
            markersize=8,
            label="Screened molecule"
        )

        # Print SMILES of screened molecules
        println("\nScreened Molecules (Year 2026):")
        for i in 1:nrow(df_mol)
            smiles = "smiles" in names(df_mol) ? df_mol[i, "smiles"] : "N/A"
            println("  det_vel=$(round(df_mol[i, "det_velocity"], digits=2)): $smiles")
        end
        println()
        text!(ax, fill(2026, nrow(df_mol)),
            df_mol[!, "det_velocity"];
            text=" (1.0)",
            align=(:left, :top),
            offset=(3, 3),
            fontsize=5
        )

    end

    # Scatter reference explosive points
    scatter!(ax,
        [p.year for p in explosive_props],
        [p.detonation_velocity for p in explosive_props];
        color=MISTStyle.UM_COLORS.maize,
        marker=:star5,
        markersize=8,
    )

    # Annotate each compound with logImpact_J
    for p in explosive_props
        text!(ax, p.year, p.detonation_velocity;
            text=string(p.compound, " (", round(p.logImpact_J; digits=1), ")"),
            align=(:left, :top),
            offset=(3, 3),
            fontsize=5
        )
    end

    scatter!(ax,
        [p.year for p in danger_props],
        [p.detonation_velocity for p in danger_props];
        color=MISTStyle.UM_COLORS.maize,
        marker=:star5,
        markersize=8,
    )

    # Annotate each compound with logImpact_J
    for p in danger_props
        text!(ax, p.year, p.detonation_velocity;
            text=string(p.compound, " (", round(p.logImpact_J; digits=1), ")"),
            align=(:left, :top),
            offset=(3, 3),
            fontsize=5
        )
    end

    # axislegend(ax; position=:lt)

    return f
end

function figure_he_screening(trace, df_mol)
    f = Figure(size=(200, 200))

    if "det_velocity" in names(df_mol) && "logImpact_J" in names(df_mol)
        pareto_kwargs = (;
            linewidth=1.5pt,
            linestyle=:solid,
            alpha=0.7,
        )

        ax = Axis(f[1, 1];
            xlabel="Detonation Velocity [m/s]",
            ylabel="log(Impact) [log(J)]"
        )

        # Scatter all points
        scatter!(ax, df_mol[!, "det_velocity"], df_mol[!, "logImpact_J"];
            color=(:blue, 0.3), marker=:circle, markersize=8, label="Screened Molecule")

        # Compute and draw Pareto front (negate x to maximize both)
        frontier = ScreeningPlots.get_pareto_front(
            -1 .* df_mol[!, "det_velocity"],
            df_mol[!, "logImpact_J"]
        )
        # Negate x coordinates back to original values
        frontier_corrected = map(frontier) do p
            Point2(-p[1], p[2])
        end

        # Print SMILES of Pareto front molecules
        println("\nPareto Front Molecules:")
        for pt in frontier_corrected
            # Find molecules matching this point (with small tolerance for floating point)
            matches = findall(i ->
                abs(df_mol[i, "det_velocity"] - pt[1]) < 0.01 &&
                abs(df_mol[i, "logImpact_J"] - pt[2]) < 0.01,
                1:nrow(df_mol)
            )
            for idx in matches
                smiles = "smiles" in names(df_mol) ? df_mol[idx, "smiles"] : "N/A"
                println("  det_vel=$(round(pt[1], digits=2)), logImpact=$(round(pt[2], digits=2)): $smiles")
            end
        end
        println()

        stairs!(ax, frontier_corrected;
            color=MISTStyle.UM_COLORS.blue,
            pareto_kwargs...,
        )

        # Add reference explosive compounds
        explosive_props = [
            (compound="TNT",   detonation_velocity=7188.5708386335755, logImpact_J=1.6),
        ]
        println(explosive_props)

        scatter!(ax,
            [p.detonation_velocity for p in explosive_props],
            [p.logImpact_J for p in explosive_props];
            color=MISTStyle.UM_COLORS.maize,
            marker=:star5,
            markersize=9,
            label="Reference explosives"
        )

        # Annotate each explosive compound
        for p in explosive_props
            text!(ax, p.detonation_velocity, p.logImpact_J;
                text=p.compound,
                align=(:left, :bottom),
                offset=(4, 4),
                fontsize=8
            )
        end

        axislegend(ax; position=:lt)
    end

    return f
end

for run_id in runs
    println("\nProcessing run: $run_id")
    path = joinpath(OUT_DIR, run_id)

    df = ScreeningPlots.load_generated_molecules(path)

    if nrow(df) == 0
        println("  Skipping - no molecules found")
        continue
    end

    println("  Loaded $(nrow(df)) molecules")

    # Skip specific run for figure_he_screening
    if run_id != "efe6ebf7-0de0-47e2-b69f-333e23c44a10"
        tr, _ = ScreeningPlots.performance_trace(joinpath(path, "screen.jsonl"))
        tr = DataFrame(tr)
        with_theme(MISTStyle.theme()) do
            figure_he_screening(tr, df)
        end |> MISTStyle.savefig("he_screening_$(run_id)")
    end

    # Only generate detonation_velocity_vs_time for specific run
    if run_id == "ac38e46d-a416-497c-9224-34e909afbe31"
        with_theme(MISTStyle.theme()) do
            detonation_velocity_vs_time(df)
        end |> MISTStyle.savefig("detonation_velocity_vs_time_$(run_id)")
    end

    println("  Saved to fig/")
end

println("\nDone!")
