using Makie
using DataFrames
using CairoMakie: CairoMakie
using StatsBase: cor, corspearman, corkendall
using Clustering: hclust
using JSON: JSON
using CategoricalArrays: categorical, levelcode, levels

include("../style.jl")
using .MISTStyle: MISTStyle, savefig, inch, pt

function plot_score_correlation!(f, df; correlation, kwargs...)
    columns = names(df)
    dist = Matrix{Float64}(undef, length(columns), length(columns))
    for I in eachindex(IndexCartesian(), dist)
        dist[I] = correlation(df[!, columns[I[1]]], df[!, columns[I[2]]])
    end

    # Order entries by clustering
    c = hclust(dist; linkage=:single, branchorder=:barjoseph)
    dist = dist[c.order, c.order]
    columns = columns[c.order]

    ticks = (eachindex(columns), columns)
    ax = Axis(f[1, 1];
        xticks=ticks, yticks=ticks,
        xticklabelrotation=0.55,
        xticklabelsize=5pt,
        yticklabelsize=5pt,
        aspect=DataAspect(),
    )
    return heatmap!(ax, dist; kwargs...)
end

function plot_auroc!(f, df)
    names = levels(df.name)
    ax = Axis(f[1, 1];
        ylabel="AUROC",
        ytickformat="{:.0%}",
        ylabelsize=6pt,
        yticks=LinearTicks(5),
        limits=(nothing, (0, 1)),
        xticks=(eachindex(names), names),
        xticklabelrotation=0.5,
        xticklabelsize=5pt,
        yticklabelsize=5pt,
    )
    h = barplot!(ax, levelcode.(df.name), df.auroc;
        dodge=levelcode.(df.dataset),
        color=levelcode.(df.dataset),
        colormap=MISTStyle.CAT_COLORS,
        colorrange=(1, 10)
    )

    elements = map(enumerate(levels(df.dataset))) do (i, dataset)
        PolyElement(;
            polycolor=h.colormap[][i],
            label=dataset,
        )
    end

    return f, elements
end

function figure_interp_surprise(; correlation=corspearman)
    # Crowdsourced
    df_crowd = DataFrame(JSON.parsefile("crowdsourced.json")["scores"])
    models = [
        "SCScore" => "SCScore",
        "SAScore" => "SAScore",
        "BR-SAScore" => "BR-SAScore",
        "ibm/MoLFormer-XL-both-10pct-smiles" => "MoLFormer",
        "models/mist-4yzwys2z-smiles-keukle" => "MIST-228M",
        "models/mist-n2dkcidc-smiles-canonical" => "MIST-ZINC",
    ]
    df_crowd = select(df_crowd, ["meanComplexity" => "Chemist", models...])

    # BA-Scorer
    df_ba = DataFrame(JSON.parsefile("ba-sascorer.json")["scores"])
    df_ba = select(df_ba, models)

    # Assembly Index
    df_ma = DataFrame(JSON.parsefile("assembly_index.json")["scores"])
    df_ma = select(df_ma, ["MA" => "Mol. Asm.", models...])

    auroc_ba = JSON.parsefile("ba-sascorer.json")["auroc"]
    auroc_crowd = JSON.parsefile("crowdsourced.json")["auroc"]
    auroc_ma = JSON.parsefile("assembly_index.json")["auroc"]

    rows = []
    for (model, name) in models
        push!(rows, (;
            name,
            crowd=auroc_crowd[model],
            ba=auroc_ba[model],
            ma=auroc_ma[model],
        ))
    end
    auroc = DataFrame(rows)
    auroc = stack(auroc, Not(:name), variable_name=:dataset, value_name=:auroc)
    auroc.dataset = replace.(auroc.dataset,
        "crowd" => "a) Sheridan et. al.",
        "ba" => "b) Chen et. al.",
        "ma" => "c) Assembly Index")
    auroc.name = categorical(auroc.name)
    auroc.dataset = categorical(auroc.dataset)


    f = Figure(;
        size=(3inch, 3.2inch),
        figure_padding=(2, 2, -10, 3),
    )
    # gl_crowd = GridLayout(f[1, 1])
    gl = GridLayout(f[3, 1:2])
    cb = Colorbar(gl[1, 1];
        vertical=false,
        flipaxis=false,
        # lowclip=:darkred,
        colorrange=(-1, 1),
        colormap=:vik10,
        tellheight=true,
        tellwidth=false,
        label=L"Spearman's $\rho$",
        labelsize=6pt,
        ticklabelsize=5pt,
        halign=:left,
        valign=:top,
    )
    kwargs = (; correlation, colorrange=cb.colorrange, colormap=cb.colormap)
    h_crowd = plot_score_correlation!(f[1, 1], df_crowd; kwargs...)
    h_ba = plot_score_correlation!(f[1, 2], df_ba; kwargs...)
    h_ma = plot_score_correlation!(f[2, 1], df_ma; kwargs...)
    h, elements = plot_auroc!(f[2, 2], auroc)
    Legend(gl[1, 2], elements, map(x -> x.label[], elements);
        labelsize=5pt,
        tellheight=true,
        tellwidth=true,
        nbanks=2,
        orientation=:horizontal,
        halign=:center,
        alignmode=Outside(),
    )
    colgap!(gl, 10)

    label_kwargs = (;
        fontsize=6pt,
        font=:bold,
        halign=:right,
        tellheight=false,
    )
    Label(f[1, 1, TopLeft()], "a)"; padding=(0, 35, -2, 0), label_kwargs...)
    Label(f[1, 2, TopLeft()], "b)"; padding=(0, 30, -2, 0), label_kwargs...)
    Label(f[2, 1, TopLeft()], "c)"; padding=(0, 35, -2, 0), label_kwargs...)
    Label(f[2, 2, TopLeft()], "d)"; padding=(0, 30, -2, 0), label_kwargs...)

    rowgap!(f.layout, 3)
    resize_to_layout!(f)

    return f
end
