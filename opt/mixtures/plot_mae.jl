
using PythonCall
using DataFrames
using Makie
using CSV: CSV
using MISTStyle
using Format: format
using JSON
using StatsBase: mean, cor, corspearman, weights, tiedrank
using JSON: JSON
using LinearAlgebra
using Mixtures: Mixtures

format_target(x::String) = replace((strip∘first∘split)(x, "["), "_" => " ")

function plot_mae!(f, df, target, units)
    df = dropmissing(df, ["$(target)_ref"])
    df.smi1 = first.(df.compounds)
    df.smi2 = last.(df.compounds)
    transform!(df,
        :smi1 => ByRow(Mixtures.label_functional_groups) => :fg1,
        :smi2 => ByRow(Mixtures.label_functional_groups) => :fg2,
    )
    transform!(df, [target, "$(target)_ref"] => ByRow(-) => target)

    fg = unique(Iterators.flatten(df.fg1))
    union!(fg, Iterators.flatten(df.fg2))
    pairs = Matrix{Float32}(undef, length(fg), length(fg))
    pair_count = Matrix{Int}(undef, length(fg), length(fg))
    fg_mae = Vector{Float32}(undef, length(fg))
    fg_count = Vector{Int}(undef, length(fg))
    for I in CartesianIndices(pairs)
        fga  = fg[I[1]]
        fgb  = fg[I[2]]
        dfs = filter(row ->
        (fga in row.fg1 && fgb in row.fg2) ||
        (fga in row.fg2 && fgb in row.fg1),
        df
        )
        pair_count[I] = nrow(dfs)
        pairs[I] = mean(abs.(dfs[!, target]))
        if I[1] == I[2]
            dfs = filter(row ->
            (fga in row.fg1) ||
            (fga in row.fg2 ),
            df
            )
            fg_mae[I[1]] = mean(abs.(dfs[!, target]))
            fg_count[I[1]]  = nrow(dfs)
        end
    end
    sdx = sortperm(fg_mae; rev=true)
    pairs = pairs[sdx, sdx]
    fg_mae = fg_mae[sdx]
    pair_count = pair_count[sdx, sdx]
    fg_count = fg_count[sdx]
    fg = fg[sdx]

    f = GridLayout(f)
    axb = Axis(f[1, 3];
        limits = ((0, nothing), nothing),
        xticks = WilkinsonTicks(5; k_min=2),
        yticks = (1:length(fg), titlecase.(fg)),
        yticklabelsize=5pt,
        xlabel = units,
    )

    ax = Axis(f[1, 4];
        yticksvisible=false,
        yticklabelsvisible=false,
        xticks = (1:length(fg), titlecase.(fg)),
        xticksvisible=true,
        xgridvisible=false,
        ygridvisible=false,
        xticklabelsvisible=true,
        xticklabelrotation = 0.785,
        backgroundcolor = :gray90,
    )
    axs = Axis(f[1, 1];
        limits = ((0, nothing), (0, nothing)),
        yticklabelsize=5pt,
        xlabel = "Component Count",
        ylabel = units,
    )
    axp = Axis(f[1, 2];
        limits = ((0, nothing), (0, nothing)),
        yticklabelsize=5pt,
        xlabel = "Mixture Count",
        yticklabelsvisible=false,
        yticksvisible=false,

    )
    # hidedecorations!(ax)
    mask = fill(NaN, length(fg), length(fg))
    triu!(mask, 1)



    cb = Colorbar(f[1, 5];
        colormap=MISTStyle.CONTINUOUS_COLORS,
        colorrange=extrema(fg_mae),
        label = units,
        vertical=true,
        tellheight=true,
        tellwidth=true,
        flip_vertical_label=true
    )
    h = heatmap!(ax, pairs + mask;
        colorscale=cb.scale,
        colormap=cb.colormap,
        colorrange= cb.colorrange,
        highclip=cb.highclip,
        lowclip=cb.lowclip,
    )
    barplot!(axb, fg_mae;
        direction = :x,
        strokewidth=0.25pt,
        strokecolor=:black,
        color=fg_mae,
        colorscale=cb.scale,
        colorrange= cb.colorrange,
        colormap=cb.colormap,
        highclip=cb.highclip,
        lowclip=cb.lowclip,
    )

    scatter!(
        axs, fg_count, fg_mae;
    )
    scatter!(
        axp, pair_count[:], pairs[:];
    )

    linkyaxes!(ax, axb)
    linkyaxes!(axs, axp)
    colgap!(f, 1, 1pt)

    return f
end


function plot_mae(df)
    targets = [
        ("molar_enthalpy_excess", L"MAE$ $(J/mol)"),
        ("density_excess", L"MAE (g/cm$^3$)"),
        ("molar_volume_excess", L"MAE (cm$^3$/mol)"),
    ]
    f = Figure(; size = (160mm, 150mm), figure_padding = (2, 2, 2, 2))
    gl = GridLayout(f[1,1])
    for (idx, (target, units)) in enumerate(targets)
        a = Int(2*idx - 1)
        b = Int(2*idx)
        Label(gl[a, :],  titlecase(format_target(target)), tellheight = true, tellwidth = false)
        plot_mae!(gl[b, :], df, target, units)
    end
    return f
end
