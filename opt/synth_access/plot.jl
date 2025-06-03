using Makie
using DataFrames
using GLM
using RegressionTables: RegressionTables, regtable, LatexTable
using StatsBase: mean, cor, corspearman, corkendall, mad, rmse
using Clustering: hclust
using JSON: JSON
using CategoricalArrays: categorical, levelcode, levels
using MISTStyle: MISTStyle, savefig, inch, pt

function plot_score_correlation!(f, df; correlation, kwargs...)
    columns = names(df)
    dist = Matrix{Float64}(undef, length(columns), length(columns))
    for I in eachindex(IndexCartesian(), dist)
        x = df[!, columns[I[1]]]
        y = df[!, columns[I[2]]]
        s = ismissing.(x) .| ismissing.(y)
        x = skipmissing(x[.!s]) |> collect
        y = skipmissing(y[.!s]) |> collect
        @assert length(x) == length(y)
        dist[I] = correlation(x, y)
    end

    # Order entries by clustering
    c = hclust(0.5 .- 0.5 .* dist; linkage=:single, branchorder=:barjoseph)
    dist = dist[c.order, c.order]
    columns = columns[c.order]

    ticks = (eachindex(columns), columns)
    ax = Axis(f[1, 1];
        xticks=ticks, yticks=ticks,
        xticklabelrotation=0.55,
        xticklabelsize=5pt,
        yticklabelsize=5pt,
        xticklabelsvisible=false,
        xticksvisible=false,
        aspect=DataAspect(),
    )
    return heatmap!(ax, dist; kwargs...)
end

function plot_auroc!(f, df)
    # Order by average score
    avg_score = combine(groupby(df, :name), :auroc => mean)
    sort!(avg_score, :auroc_mean; rev=true)
    df.name = categorical(string.(df.name); levels=avg_score.name)
    min_auroc = minimum(df.auroc)
    min_auroc = min_auroc < 0.5 ? 0 : 0.5

    names = levels(df.name)
    ax = Axis(f[1, 1];
        ylabel="AUROC",
        ytickformat="{:.0%}",
        ylabelsize=6pt,
        yticks=LinearTicks(5),
        limits=(nothing, (min_auroc, 1)),
        xticks=(eachindex(names), names),
        xticklabelrotation=0.5,
        xticklabelsize=5pt,
        yticklabelsize=5pt,
    )
    gini = @. 2 * df.auroc - 1
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

function load_scores()
    # Crowdsourced
    df_crowd = DataFrame(JSON.parsefile("crowdsourced.json")["scores"])
    df_crowd_ma = DataFrame(JSON.parsefile("crowdsourced-asm.json")["scores"])
    leftjoin!(df_crowd, df_crowd_ma; on=["is_complex", "smiles", "smiles-canonical", "smiles-kekule", "stdevComplexity", "meanComplexity"])
    replace!(df_crowd[!, "assembly-index"], NaN => missing)

    auroc_crowd = JSON.parsefile("crowdsourced.json")["auroc"]
    auroc_crowd_ma = JSON.parsefile("crowdsourced-asm.json")["auroc"]
    auroc_crowd = merge(auroc_crowd, auroc_crowd_ma)

    # BA-Scorer
    df_ba = DataFrame(JSON.parsefile("ba-sascorer.json")["scores"])
    df_ba_ma = DataFrame(JSON.parsefile("ba-sascorer-asm.json")["scores"])
    leftjoin!(df_ba, df_ba_ma; on=["is_hard", "smiles", "smiles-canonical", "smiles-kekule"])
    replace!(df_ba[!, "assembly-index"], NaN => missing)

    auroc_ba = JSON.parsefile("ba-sascorer.json")["auroc"]
    auroc_ba_ma = JSON.parsefile("ba-sascorer-asm.json")["auroc"]
    auroc_ba = merge(auroc_ba, auroc_ba_ma)

    return (;
        crowd = (; score = df_crowd, auroc = auroc_crowd),
        ba = (; score = df_ba, auroc = auroc_ba),
    )
end

function model_auroc(auroc::Dict)
    df = stack(DataFrame(auroc); variable_name=:model, value_name=:auroc)
    subset!(df, :model => ByRow(x -> occursin("/", x)))
    @. df.untrained = occursin("untrained", df.model)
    @. df.per_token = occursin("per-token", df.model)
    df.encoding = map(df.model) do m
        if occursin("smiles-canonical", m)
            return "canonical"
        elseif occursin("smiles-kekule", m)
            return "kekule"
        else
            return "smiles"
        end
    end
    df.encoding .= categorical(df.encoding)
    df.model = map(df.model) do m
        m = replace(m, "smiles-canonical" => "", "smiles-kekule" => "", "per-token" => "", "untrained" => "", "smiles" => "")
        m = replace(m, r"-+$" => "")
    end

    # Fit linear model
    lm(
        @formula(auroc ~ untrained + per_token + encoding + model),
        df;
        contrasts = Dict(
            :untrained => EffectsCoding(; base=false),
            :per_token => EffectsCoding(; base=false),
            :encoding => EffectsCoding(; base="smiles"),
            :model => EffectsCoding(; base="models/mist-ti624ev1"),
        )
    )
end

function figure_interp_surprise(models; correlation=corspearman, size=(3.42inch, 1.0inch))
    # Crowdsourced
    o = load_scores()
    df_crowd = select(o.crowd.score, ["meanComplexity" => "Chemist", models...])
    df_ba = select(o.ba.score, models)

    auroc_crowd = o.crowd.auroc
    auroc_ba = o.ba.auroc

    rows = []
    for (model, name) in models
        push!(rows, (;
            name,
            crowd=auroc_crowd[model],
            ba=auroc_ba[model],
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


    f = Figure(; size, figure_padding=(2, 2, 2, 3))
    gl = GridLayout(f[1, 1])
    if correlation == cor
        cb_label = L"Pearson's $\rho$"
    elseif correlation == corspearman
        cb_label = L"Spearman's $\rho$"
    else
        cb_label = string(correlation)
    end
    cb = Colorbar(gl[2, 1:2];
        vertical=false,
        flipaxis=false,
        # lowclip=:darkred,
        colorrange=(-1, 1),
        colormap=:vik10,
        tellheight=true,
        tellwidth=false,
        label=cb_label,
        labelsize=6pt,
        ticklabelsize=5pt,
        halign=:left,
        valign=:top,
    )
    kwargs = (; correlation, colorrange=cb.colorrange, colormap=cb.colormap)
    h_crowd = plot_score_correlation!(gl[1, 1], df_crowd; kwargs...)
    h_ba = plot_score_correlation!(gl[1, 2], df_ba; kwargs...)
    h, elements = plot_auroc!(f[1, 2], auroc)
    Legend(f[1, 2], elements, MISTStyle.label.(elements);
        # tellheight=true,
        # tellwidth=true,
        nbanks=1,
        padding=(1, 1, 1, 1),
        margin=(1, 1, 1, 1),
        valign=:top,
        halign=:right,
        # alignmode=Inside(),
    )
    colgap!(gl, 10)

    label_kwargs = (;
        fontsize=6pt,
        font=:bold,
        halign=:right,
        tellheight=false,
    )
    Label(gl[1, 1, TopLeft()], "a)"; padding=(0, 30, -2, 0), label_kwargs...)
    Label(gl[1, 2, TopLeft()], "b)"; padding=(0, 30, -2, 0), label_kwargs...)
    Label(f[1, 2, TopLeft()], "c)"; padding=(0, 20, -2, 0), label_kwargs...)
    # Label(f[2, 2, TopLeft()], "d)"; padding=(0, 30, -2, 0), label_kwargs...)
    colsize!(f.layout, 1, Auto(2))

    rowgap!(f.layout, 3)
    resize_to_layout!(f)

    return f
end

function create_figures()
    main_models = [
        "SCScore" => "SCScore",
        "SAScore" => "SAScore",
        "smirk" => "Smirk Fertility",
        "ibm-research/MoLFormer-XL-both-10pct-smiles-per-token" => "MoLFormer",
        "seyonec/ChemBERTa-zinc-base-v1-smiles-per-token" => "ChemBERTa",
        "models/mist-ti624ev1-smiles-kekule-per-token" => "MIST-27M",
        "models/mist-4yzwys2z-smiles-kekule-per-token" => "MIST-228M",
        "models/mist-1.8B-dh61satt-smiles-kekule-per-token" => "MIST-1.8B",
        "models/mist-1.8B-dh61satt-untrained-smiles-kekule-per-token" => "MIST-1.8B, Untrained",
        "models/mist-n2dkcidc-smiles-canonical-per-token" => "MIST-ZINC",
        "assembly-index" => "Mol. Asm.",
    ]
    all_models = [
        "molecular-weight" => "Mol. Weight",
        "BR-SAScore" => "BR-SAScore",
        "ibm-research/MoLFormer-XL-both-10pct-untrained-smiles-per-token" => "MoLFormer, Untrained",
        "ibm-research/MoLFormer-XL-both-10pct-smiles-kekule-per-token" => "MoLFormer, Kekule",
        "ibm-research/MoLFormer-XL-both-10pct-smiles-canonical-per-token" => "MoLFormer, Canonical",
        "seyonec/ChemBERTa-zinc-base-v1-untrained-smiles-per-token" => "ChemBERTa, Untrained",
        "seyonec/ChemBERTa-zinc-base-v1-smiles-kekule-per-token" => "ChemBERTa, Kekule",
        "seyonec/ChemBERTa-zinc-base-v1-smiles-canonical-per-token" => "ChemBERTa, Canonical",
    ]
    all_models = vcat(main_models, all_models)

    # Effect Model
    o = load_scores()
    m_crowd = model_auroc(o.crowd.auroc)
    m_ba = model_auroc(o.ba.auroc)
    regtable(m_crowd, m_ba;
        render = LatexTable(),
        file=joinpath(@__DIR__, "fig", "interp_surprise.tex"),
        regression_statistics=[
             RegressionTables.Nobs,
             RegressionTables.DOF,
             RegressionTables.R2,
             (m -> mad(residuals(m))) => "MAE",
             (m -> rmsd(predict(m), response(m))) => "RMSE",
        ]
    )

    size_large = (4.5inch, 1.5inch)
    with_theme(MISTStyle.theme()) do
        figure_interp_surprise(main_models) |> savefig("interp_surprise")
        figure_interp_surprise(main_models; correlation=cor) |> savefig("interp_surprise_cor")
        figure_interp_surprise(all_models; size=size_large) |> savefig("interp_surprise_all")
        figure_interp_surprise(all_models; correlation=cor, size=size_large) |> savefig("interp_surprise_all_cor")
    end
end
