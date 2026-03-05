using DataFrames
using Makie
using CSV: CSV
using MISTStyle
using StatsBase: mean, cor, corspearman, weights, tiedrank, countmap
using JSON: JSON
using Interpolations
using Format: format
using CategoricalArrays
using HDF5
using LinearAlgebra

# Directory containing the data release files
const DATA_DIR = realpath("/Users/anoushka/VSCodeProjects/electrolyte-fm/opt/olfactory")

function plot_count_corr!(f)
    df = DataFrame(CSV.File(joinpath(DATA_DIR, "dataset_size_auroc.csv")))
    ax = Axis(f[1, 1];
        limits=((30, 1800), (0, 1)),
        xlabel="Number of Samples",
        xticks = [100, 1000],
        xscale=log10,
        ylabel="AUROC",
        tellwidth=true
    )
    scatter!(ax, df.num_samples, df.auroc;
        color = (MISTStyle.UM_COLORS.blue, 0.5), marker = :circle,
        strokewidth = 0.25pt,
        strokecolor = :black,
        markersize = 4pt,
    )
    return f
end

function plot_functional_groups!(f)
    functional_groups = [
        "alkane", "alkene", "alkyne",
        "arene", "alcohol", "ether",
        "carbonate ester", "epoxide",
        "haloalkane", "aldehyde", "ketone",
        "carboxylic acid", "acid anhydride",
        "ester", "amide", "acyl halide",
        "amine", "nitrile", "imine",
        "isocyanate", "azo compound",
        "thiol", "aqueous", "salt"
    ]
    df = DataFrame(CSV.File(joinpath(DATA_DIR, "functional_group_errors.csv")))
    mean_error_count = Dict()
    for func_group in functional_groups
        _df = subset(df, :functional_group => ByRow(fg -> occursin(func_group, fg)))
        count_ = length(_df[!, :functional_group])
        if count_> 0
            group_auroc = mean(_df[!, :auroc])
            mean_error_count[func_group] = (group_auroc, count_)
        end
    end
    ticklabels = collect(keys(mean_error_count))
    labels = categorical(ticklabels)
    ax = Axis(f[1, 1];
        limits=((0, length(labels) + 1), (0, 1)),
        ylabel="Mean AUROC",
        xgridvisible = false,
        xticklabelrotation=0.9,
        xticklabelsize=5pt,
        xticks=(1:length(ticklabels), titlecase.(ticklabels)),
        tellwidth=true
    )
    barplot!(ax, levelcode.(labels), first.(collect(values(mean_error_count))); alpha = 0.9)

    ax_sc = Axis(f[1, 2];
        limits=((1, 2000), (0, 1)),
        xticks= [1, 10, 100, 1000],
        xlabel="Number of Molecules",
        ylabel="Mean AUROC",
        xscale=log10,
    )
    scatter!(ax_sc, last.(collect(values(mean_error_count))), first.(collect(values(mean_error_count)));
        color = (MISTStyle.UM_COLORS.blue, 0.5), marker = :circle,
        strokewidth = 0.25pt,
        strokecolor = :black,
        markersize = 4pt,
    )
    return f
end

function plot_small_change!(f)
    df = DataFrame(CSV.File(joinpath(DATA_DIR, "small_change_normalized_logits.csv")))
    cand = intersect(names(df)[2:end], names(df, Number))
    to_drop = [c for c in cand if sum(skipmissing(df[!, c])) < 0.06]
    select!(df, Not(to_drop))
    labels = categorical(names(df)[2: end])
    tickslabels = titlecase.(names(df)[2: end])
    ax = Axis(f[1, 1:2];
        limits=((0, length(tickslabels) + 1), (0, 1)),
        ylabel="Probability",
        xgridvisible = false,
        xticklabelrotation=0.9,
        xticklabelpad=0,
        xticklabelsize=5pt,
        xticks=(1:length(tickslabels), tickslabels),
        tellwidth=true
    )
    for row in eachrow(df)
        logits = collect( (row[c] for c in names(df)[2:end]) )
        barplot!(ax, levelcode.(labels), logits; label=row["Column1"])
    end
    axislegend(ax, position=:lt, padding=(1, 1, 1, 1), margin=(1, 1, 1, 1))
    return f
end

function load_matrix(matrix_name)
    return h5read(joinpath(DATA_DIR, "$matrix_name.h5"), matrix_name)
end

function plot_jaccard!(f)
    Jd = load_matrix("jaccard_data") - load_matrix("jaccard_model")
    h = hclust(Jd; linkage=:single, branchorder=:optimal)
    Jd = Jd[h.order, h.order]
    mask = fill(NaN, length(h.order), length(h.order))
    tril!(mask, -1)
    ax = Axis(f[1, 1];
        aspect=1,
        xticksvisible=false,
        yticksvisible=false,
        xticklabelsvisible=false,
        yticklabelsvisible=false,
        tellwidth=true,
    )
    hidedecorations!(ax)
    h = heatmap!(ax, Jd + mask; colormap=:lipari, colorrange=(0, maximum(Jd)))
    Colorbar(f[1, 2];
        tickformat="{:.0%}",
        ticks=LinearTicks(6),
        label=L"$\Delta$ Jaccard Index",
        colormap=h.colormap,
        colorrange=h.colorrange,
        tellwidth=true
    )
    return f
end

function plot_phi!(f)
    Jd = load_matrix("phi_data")
    h = hclust(Jd; linkage=:single, branchorder=:optimal)
    Jd = Jd[h.order, h.order]
    mask = fill(NaN, length(h.order), length(h.order))
    tril!(mask, 1)
    ax = Axis(f[1, 1];
        aspect=1,
        xticksvisible=false,
        yticksvisible=false,
        xticklabelsvisible=false,
        yticklabelsvisible=false,
        tellwidth=true,
        tellheight=true,
    )
    hidedecorations!(ax)
    h = heatmap!(ax, Jd + mask; colormap=:lipari, colorrange=(0, maximum(Jd)))
    Colorbar(f[1, 2];
        ticks=LinearTicks(6),
        label=L"$\phi_{model} - \phi_{data}$",
        colormap=h.colormap,
        colorrange=h.colorrange,
        tellwidth=true
    )
    return f
end

function columnwise_cosine_sim(W::AbstractMatrix)
    W = W[1:end - 1, :]
    @info size(W)
    colnorms = map(norm, eachcol(W))
    colnorms = sqrt.(colnorms)
    denom = colnorms' .* colnorms     # pairwise norm products
    S = (W' * W) ./ denom
    @info size(S)
    S[.!isfinite.(S)] .= 0.0          # handle zero-norm columns
    return S
end

function plot_logits!(f)
    df = DataFrame(CSV.File(joinpath(DATA_DIR, "raw_prediction.csv")))
    select!(df, names(df)[1:40])
    labels = categorical(names(df)[2: end])
    tickslabels = titlecase.(names(df)[2: end])
    L = Matrix(df[:, names(df)[2:end - 1]])
    C = cor(L)
    h = hclust(C; linkage=:single, branchorder=:barjoseph)
    C = C[h.order, h.order]
    tickslabels = tickslabels[h.order]
    ax = Axis(
        f[2, 1];
        aspect=1,
        yticks=(1:length(tickslabels), tickslabels),
        xticks=(1:length(tickslabels), tickslabels),
        yticksvisible=false,
        xticksvisible=false,
        xticklabelrotation=0.5*pi,
        xticklabelsize=4pt,
        yticklabelsvisible=false
    )
    # hidedecorations!(ax)
    h = heatmap!(ax, C; colormap=:vik, colorrange=(-1, 1))
    Colorbar(f[1, 1];
        ticks=LinearTicks(6),
        label="Correlation",
        colormap=h.colormap,
        colorrange=h.colorrange,
        vertical=false,
        tellwidth=true,
        tellheight=true
    )
    return f
end

function plot_features!(f)
    features = load_matrix("weight")
    S = columnwise_cosine_sim(features)
    h = hclust(S; linkage=:average, branchorder=:barjoseph)
    S = S[h.order, h.order]
    ax = Axis(f[1, 1];
        aspect=1,
    )
    S = triu(S, 1)
    hidedecorations!(ax)
    h = heatmap!(ax, S; colormap=:vik, colorrange=(-0.1, 0.1))
    Colorbar(f[1, 2];
        ticks=LinearTicks(6),
        label="Feature Cos. Sim.",
        colormap=h.colormap,
        colorrange=h.colorrange,
        tellwidth=true
    )
    return f
end


function olfactory_panel()
    f = Figure(; size=(200mm, 72mm), figure_padding=(5, 5, 5, 5))

    gl = GridLayout(f[1, 1:3])
    plot_count_corr!(gl[1, 1])
    plot_functional_groups!(gl[1, 2:3])
    plot_small_change!(f[2, 1:3])

    # plot_phi!(f[1, 4])
    plot_logits!(f[1:2, 4])

    sublabel!(gl[1, 1, TopLeft()], "a"; left=15pt)
    sublabel!(gl[1, 2, TopLeft()], "b"; left=15pt)
    sublabel!(gl[1, 3, TopLeft()], "c"; left=15pt)
    sublabel!(f[2, 1, TopLeft()], "d"; left=14pt)
    sublabel!(f[1, 4, TopLeft()], "e"; left=20pt)

    colgap!(f.layout, 6pt)
    resize_to_layout!(f)

    return f
end
