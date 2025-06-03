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
        yticks=LinearTicks(5),
        limits=(nothing, (min_auroc, 1)),
        xticks=(eachindex(names), names),
        xticklabelrotation=0.6,
    )
    gini = @. 2 * df.auroc - 1
    h = barplot!(ax, levelcode.(df.name), df.auroc)

    return f
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

function figure_interp_surprise(score, auroc, models; correlation=corspearman, size=(3.42inch, 1.76inch))
    rows = []
    for (model, name) in models
        if haskey(auroc, model)
            push!(rows, (;
                name,
                auroc=auroc[model],
            ))
        end
    end
    score = select(score, models)
    auroc = DataFrame(rows)
    auroc.name = categorical(auroc.name)

    f = Figure(; size, figure_padding=(2, 2, 2, 6))
    gl = GridLayout(f[1, 1])
    if correlation == cor
        cb_label = L"Pearson's $\rho$"
    elseif correlation == corspearman
        cb_label = L"Spearman's $\rho$"
    else
        cb_label = string(correlation)
    end
    cb = Colorbar(gl[2, 1];
        vertical=false,
        flipaxis=false,
        colorrange=(-1, 1),
        colormap=:vik10,
        tellheight=true,
        tellwidth=true,
        label=cb_label,
        halign=:left,
        valign=:top,
    )
    kwargs = (; correlation, colorrange=cb.colorrange, colormap=cb.colormap)
    h_crowd = plot_score_correlation!(gl[1, 1], score; kwargs...)
    h = plot_auroc!(f[1, 2], auroc)
    colgap!(gl, 3)

    label_kwargs = (;
        fontsize=8pt,
        font=:bold,
        halign=:right,
        tellheight=false,
    )
    Label(gl[1, 1, TopLeft()], "a)"; padding=(0, 30, 5, 0), label_kwargs...)
    Label(f[1, 2, TopLeft()], "b)"; padding=(0, 20, 5, 0), label_kwargs...)

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

    size_large = (4.5inch, 2.3inch)
    with_theme(MISTStyle.theme()) do
        for (name, ds) in ["crowd" => o.crowd, "ba" => o.ba]
            if name == "crowd"
                models = ["meanComplexity" => "Chemist", main_models...]
            else
                models = main_models
            end
            figure_interp_surprise(ds.score, ds.auroc, models) |> savefig("interp_surprise_$name")
            figure_interp_surprise(ds.score, ds.auroc, models; correlation=cor) |> savefig("interp_surprise_cor_$name")
            figure_interp_surprise(ds.score, ds.auroc, vcat(models, all_models); size=size_large) |> savefig("interp_surprise_all_$name")
            figure_interp_surprise(ds.score, ds.auroc, vcat(models, all_models); correlation=cor, size=size_large) |> savefig("interp_surprise_all_cor_$name")
        end
    end
end
