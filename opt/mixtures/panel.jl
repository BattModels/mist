using CairoMakie: CairoMakie
using MISTStyle
using DataFrames
using CSV: CSV
using Interpolations: linear_interpolation
using StatsBase: mad
using LaTeXStrings

function mixture_panel()
    df_excess = DataFrame(CSV.File("panel/excess/parity_diqlnysx.csv"))
    df_enthalpy = DataFrame(CSV.File("panel/enthalpy/parity_ulqfugc4.csv"))
    df_ionic = DataFrame(CSV.File("panel/ionic/parity_w8phe03q.csv"))
    filter!(row -> row.predicted < 5 , df_ionic)
    filter!(row -> row.predicted > -5 , df_ionic)
    df_list = [df_excess,df_enthalpy, df_ionic]

    # Create figure with desired physical
    fig = Figure(figure_padding = 10; size=(4.4606299inch, 3.50394inch))

    scatter_xlabel = [L"\textrm{Actual }[cm^3/mol]",L"\textrm{Actual }[J/mol]", L"\textrm{Actual }[mS/cm]"]
    scatter_ylabel = [L"\textrm{MIST }[cm^3/mol]", L"\textrm{MIST }[J/mol]", L"\textrm{MIST }[mS/cm]"]

    for col in 1:3
        df = df_list[col]
        ax = Axis(fig[1, col];
            xlabel=scatter_xlabel[col],
            ylabel=scatter_ylabel[col],
        )
        lines!(ax, df.actual, df.actual;
            color=:black, linestyle=:solid

        )
        scatter!(ax, df.actual, df.predicted;
            color=df.temperature
        )
        # errorbars!(ax, df.actual, df.predicted, df.st_dev; color = df.temperature)
    end
    cb = Colorbar(fig[3,:];
        label = L"Temperature [K]$$",
        colormap = MISTStyle.CONTINUOUS_COLORS,
        colorrange = (240, 340),
        flipaxis = false,
        vertical=false
    )
    plot_excess_molar_volume!(fig, cb)
    plot_excess_molar_enthalpy!(fig, cb)
    plot_ionic_conductivity!(fig, cb)

    return fig
end


using CairoMakie, CSV, DataFrames
using MISTStyle  # for MISTStyle.cb_attrs

function plot_excess_molar_volume!(fig, cb::Colorbar)
    # Load data
    df     = DataFrame(CSV.File("panel/excess/inference.csv"))
    df_ref = DataFrame(CSV.File("panel/excess/test.csv"))

    # Create axis
    ax = Axis(fig[2, 1];
        xlabel              = L"x_1",
        ylabel              = L"V_m [cm^3/mol]",
        xlabelvisible       = true,
        xticksvisible       = true,
        xticklabelsvisible  = true,
        limits = ((nothing, nothing), (-1.1, 0.2))
    )

    # Define a distinct marker for each (smi1, smi2) pair
    pairs = collect(Iterators.product(unique(df.smi1), unique(df.smi2)))
    marker_list = [:circle, :rect, :diamond, :utriangle, :dtriangle, :hexagon, :cross, :xcross]
    marker_map = Dict(p => marker_list[mod1(i, length(marker_list))] for (i, p) in enumerate(pairs))
    temp_count = 0
    # Plot lines and scatter per temperature, per pair
    for (smi1, smi2) in pairs

        # filter inference and reference
        sub_df     = filter(row -> row.smi1 == smi1 && row.smi2 == smi2, df)
        sub_df_ref = filter(row -> row.smi1 == smi1 && row.smi2 == smi2, df_ref)

        mk = marker_map[(smi1, smi2)]
        if temp_count%7 == 0
        for g in groupby(sub_df, :temperature)
            lines!(ax, g.x1, g.predicted_mean;
                color = g.temperature,
                MISTStyle.cb_attrs(cb, Lines)...,
            )
        end
        for g in groupby(sub_df_ref, :temperature)
                h = scatter!(ax, g.x1, g[!,"excess_molar_volume/(cm3/mol)"];
                color      = g.temperature,
                marker     = mk,
                    label      = "$(smi1) & $(smi2)",
                MISTStyle.cb_attrs(cb, Lines)...,
            )

            end
            temp_count += 1
        else
            temp_count += 1
            continue
        end
    end
    axislegend(ax, merge = true, unique = true, position = :rb, nbanks=1, padding=(1, 1, 1, 1),
    margin=(1, 1, 1, 1))
    return fig
end

function plot_excess_molar_enthalpy!(fig, cb::Colorbar)
    # Load data
    df     = DataFrame(CSV.File("panel/enthalpy/inference.csv"))
    df_ref = DataFrame(CSV.File("panel/enthalpy/test.csv"))

    # Create axis
    ax = Axis(fig[2, 2];
        xlabel              = L"x_1",
        ylabel              = L"H_m [cm^3/mol]",
        xlabelvisible       = true,
        xticksvisible       = true,
        xticklabelsvisible  = true,
        limits = ((nothing, nothing), (-1000, nothing))
    )

    # Define a distinct marker for each (smi1, smi2) pair
    pairs = collect(Iterators.product(unique(df.smi1), unique(df.smi2)[1:10]))
    marker_list = [:circle, :star5, :rect, :diamond, :ltriangle, :utriangle, :hline, :rtriangle,:vline, :star4, :pentagon, :dtriangle, :hexagon, :cross, :xcross]
    marker_map = Dict(p => marker_list[mod1(i, length(marker_list))] for (i, p) in enumerate(pairs))
    temp_count = 0
    # Plot lines and scatter per temperature, per pair
    for (smi1, smi2) in pairs

        # filter inference and reference
        sub_df     = filter(row -> row.smi1 == smi1 && row.smi2 == smi2, df)
        sub_df_ref = filter(row -> row.smi1 == smi1 && row.smi2 == smi2, df_ref)

        mk = marker_map[(smi1, smi2)]
        if temp_count%1 == 0
        for g in groupby(sub_df, :temperature)
            lines!(ax, g.x1, g.predicted_mean;
                color = g.temperature,
                MISTStyle.cb_attrs(cb, Lines)...,
            )
        end
        for g in groupby(sub_df_ref, :temperature)
                h = scatter!(ax, g.x1, g[!,"excess_molar_enthalpy/(J/mol)"];
                color      = g.temperature,
                marker     = mk,
                    label      = "$(smi2)",
                MISTStyle.cb_attrs(cb, Lines)...,
            )

            end
            temp_count += 1
        else
            temp_count += 1
            continue
        end
    end
    axislegend(ax, merge = true, unique = true, position = :rb, nbanks=2, padding=(1, 1, 1, 1),
    margin=(1, 1, 1, 1))
    return fig
end

function plot_ionic_conductivity!(fig, cb::Colorbar)
    csv_path = "panel/ionic/ionic_conductivity_curves.csv"
    # Read data; expect columns: solvent, x_Li, σ (or sigma), temperature
    df = CSV.read(csv_path, DataFrame)

    # List the two salts (panels) in order
    salts = ["LiTFSI" ] # unique(df.salt_name)

    # Sort temperatures and build a continuous, dark‐cropped colormap
    temps = sort(unique(df.temperature))
    axes = [
        Axis(fig[1 + i, 3];
            xlabel           = L"x_{Li}",    # only show x‐label on bottom panels
            ylabel           = L"\sigma\ [mS/cm]",
            # xticks           = (0:0.05:0.2, ["0.0", "0.05","0.1","0.15","0.2"]),
            # yticks           = (0:2:6, ["0","2","4","6"]),
        )
        for (i, salt) in enumerate(salts)
    ]

    # Plot each temperature curve in both panels
    for ax in axes, T in temps
        sol = "LiTFSI"
        mask = (df.salt_name .== sol) .& (df.temperature .== T)
        sub = df[mask, :]
        lines!(
            ax,
            sub.composition,
            sub.predictions;
            linewidth=0.5pt,
            color     = sub.temperature,
            MISTStyle.cb_attrs(cb, Lines)...
        )
    end
    fig
end
