#!/usr/bin/env -S julia --project=@script --startup-file=no
using Makie
using DataFrames
using GLM
using RegressionTables: RegressionTables, regtable, LatexTable
using StatsBase: mean, cor, corspearman, corkendall, mad, rmse
using Clustering: hclust
using JSON: JSON
using CategoricalArrays: categorical, levelcode, levels
using MISTStyle: MISTStyle, savefig, sublabel!, inch, pt
using LinearAlgebra: dot, norm

cosine_similarity(x, y) = dot(x, y) / (norm(x) * norm(y))
cosine_distance(x, y) = 1 - cosine_similarity(x, y)

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

function align_scores(left::DataFrame, right::DataFrame, source)
    # Each task scores the same materialized rows. Joining on all common
    # columns can silently discard conflicting/missing rows or multiply
    # duplicates, changing the cohort used in the paper's comparisons.
    for data in (left, right)
        "row_id" in names(data) || error("Missing row_id in $source")
        all(id -> id isa Integer && !(id isa Bool) && id >= 0, data.row_id) ||
            error("Invalid row_id in $source")
        allunique(data.row_id) || error("Duplicate row_id in $source")
    end
    isempty(left) && error("Empty score cohort in $source")
    Set(left.row_id) == Set(right.row_id) || error("Score cohort differs in $source")
    lookup = Dict(id => index for (index, id) in enumerate(right.row_id))
    right = right[[lookup[id] for id in left.row_id], :]
    common = intersect(names(left), names(right))
    for column in common
        isequal(left[!, column], right[!, column]) ||
            error("Conflicting $column values in $source")
    end
    return hcat(left, select(right, Not(common)))
end

function load_scores(model_dirs, heuristics_dir, baseline_dirs=String[])
    function merged(dataset)
        model = JSON.parsefile(joinpath(first(model_dirs), dataset * ".json"))
        heuristic = JSON.parsefile(joinpath(heuristics_dir, dataset, "scores.json"))
        left, right = DataFrame(model["scores"]), DataFrame(heuristic["scores"])
        df = align_scores(left, right, dataset * " heuristics")
        auroc = merge(model["auroc"], heuristic["auroc"])
        for directory in vcat(model_dirs[2:end], baseline_dirs)
            baseline = JSON.parsefile(joinpath(directory, dataset * ".json"))
            scores = DataFrame(baseline["scores"])
            df = align_scores(df, scores, dataset * " model " * directory)
            merge!(auroc, baseline["auroc"])
        end
        for col in names(df)
            df[!, col] = [isnothing(value) ? missing : value for value in df[!, col]]
        end
        return (; score=df, auroc)
    end
    return (; crowd=merged("crowdsourced"), ba=merged("ba-sascorer"))
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

function model_score_grid(df::DataFrame, baseline::String; cmax=100)
    f = Figure(; size=(7inch, 6inch), figure_padding=(5, 5, 1, 1))
    df = subset(df, baseline => ByRow(!ismissing))
    baseline_score = df[!, baseline]
    df = select(df, Not(baseline))
    models = names(df)
    models = filter(models) do m
        m in ["row_id", "stdevComplexity", "is_complex", "is_hard"] && return false
        return eltype(df[!, m]) <: Union{Integer, AbstractFloat, Union{Missing, <:AbstractFloat}}
    end
    models = DataFrame(map(label_model, models))
    sort!(models, [order(:model, rev=false), :per_token, :encoding, :untrained]; rev=true)
    plt_names = model_plot_names()

    cb = Colorbar(f[3, 1];
        colorrange=(1, cmax),
        scale=log10,
        vertical=false,
        flipaxis=false,
        label="Number of Molecules"
    )
    ax_kwargs = (;
        ylabelvisible=false,
        yticksvisible=false,
        yticklabelsvisible=false,
        xlabelvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
    )
    scatter_kwargs = (;
        marker=:circle,
        alpha=0.2,
        color=MISTStyle.UM_COLORS.blue,
    )
    hexbin_kwargs = (;
        bins=40,
        colorscale=cb.scale,
        colorrange=cb.colorrange,
        colormap=cb.colormap,
    )
    plt_kwargs = hexbin_kwargs



    # Heuristics
    gl = GridLayout(f[1, 1])
    heuritics = subset(models, :name => ByRow(n -> !occursin("/", n)))
    for (i, heuristic) in enumerate(eachrow(heuritics))
        ax = Axis(gl[1, i];
            title=get(plt_names, heuristic.name, heuristic.name),
            ax_kwargs...,
        )
        x = df[!, heuristic.name]
        valid = (!ismissing).(x)
        hexbin!(ax, x[valid], baseline_score[valid]; plt_kwargs...)
    end

    mfm = subset(models, :model => ByRow(m -> occursin("/", m)))
    mfm_models = unique(mfm.model)
    nrow = length(mfm_models)
    ncol = cld(size(mfm, 1), nrow)
    gl = GridLayout(f[2, 1], nrow, ncol)
    for (m, mfm_name) in enumerate(mfm_models)
        variants = subset(models, :model => ByRow(==(mfm_name)))
        if !haskey(plt_names, mfm_name)
            @warn "No plot name for $mfm_name"
        end
        for (j, variant) in enumerate(eachrow(variants))
            ax = Axis(gl[m, j];
                ylabel=get(plt_names, mfm_name, mfm_name),
                ylabelfont=:bold,
                ax_kwargs...,
            )
            if j == 1
                ax.ylabelvisible[] = true
            end
            if m == 1
                ax.title[] = label_varient(variant)
            end
            hexbin!(ax, df[!, variant.name], baseline_score; plt_kwargs...)
        end
    end

    sublabel!(f[1, 1, TopLeft()], "a")
    sublabel!(f[2, 1, TopLeft()], "b")
    rowsize!(f.layout, 2, Auto(nrow))
    resize_to_layout!(f)

    return f
end

function label_varient(variant)
    ut = variant.untrained ? "Untrained" : "Trained"
    norm = variant.per_token ? "Per-Token" : "Per-Molecule"
    encoding = Dict(
        "smiles" => "Smiles",
        "kekule" => "Kekule",
        "canonical" => "Canonical",
    )[variant.encoding]
    return "$ut\n$norm\n$encoding"
end

function model_plot_names()
    return Dict(
        "mist/mist-28m" => "MIST-28M",
        "mist/mist-1p8b" => "MIST-1.8B",
        "seyonec/ChemBERTa-zinc-base-v1" => "ChemBERTa",
        "models/mist-ti624ev1" => "MIST-27M",
        "models/mist-4yzwys2z" => "MIST-228M",
        "models/mist-1.8B-dh61satt" => "MIST-1.8B",
        "models/mist-n2dkcidc" => "MIST-ZINC",
        "models/mist-28znv46w" => "MIST-128M",
        "ibm-research/MoLFormer-XL-both-10pct" => "MoLFormer",
        "molecular-weight" => "Molecular Weight",
        "assembly-index" => "Molecular Assembly",
        "smirk" => "Smirk",
        "smirk-kekule" => "Smirk, Kekule",
        "smirk-canonical" => "Smirk, Canonical",
    )
end

function label_model(name::String)
    untrained = occursin("untrained", name)
    per_token = occursin("per-token", name)
    if occursin("smiles-canonical", name)
        encoding = "canonical"
    elseif occursin("smiles-kekule", name)
        encoding = "kekule"
    else
        encoding = "smiles"
    end
    model = replace(name, "smiles-canonical" => "", "smiles-kekule" => "", "per-token" => "", "untrained" => "", "smiles" => "")
    model = replace(model, r"-+$" => "")
    return (; model, untrained, per_token, encoding, name)
end


function model_auroc(auroc::Dict; base_model="models/mist-ti624ev1")
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

    contrasts = Dict(
            :untrained => EffectsCoding(; base=false),
            :per_token => EffectsCoding(; base=false),
            :encoding => EffectsCoding(; base="smiles"),
    )
    if length(unique(df.model)) > 1
        contrasts[:model] = EffectsCoding(; base=base_model)
        return lm(@formula(auroc ~ untrained + per_token + encoding + model), df; contrasts)
    end
    return lm(@formula(auroc ~ untrained + per_token + encoding), df; contrasts)
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

function create_figures(bases, model_dirs, heuristics_dir, output_dir; baseline_dirs=String[])
    length(bases) == length(model_dirs) > 0 || error("Provide one --model-dir for each --base")
    main_models = [
        "SCScore" => "SCScore",
        "SAScore" => "SAScore",
        "smirk" => "Smirk",
        "ibm-research/MoLFormer-XL-both-10pct-smiles" => "MoLFormer",
        "seyonec/ChemBERTa-zinc-base-v1-smiles" => "ChemBERTa",
    ]
    for base in bases
        label = model_plot_names()["mist/$base"]
        append!(main_models, [
            "mist/$base-smiles-kekule" => label,
            "mist/$base-untrained-smiles-kekule" => label * ", Untrained",
        ])
    end
    push!(main_models, "assembly-index" => "Molecular Assembly")
    all_models = [
        "molecular-weight" => "Molecular Weight",
        "BR-SAScore" => "BR-SAScore",
        "ibm-research/MoLFormer-XL-both-10pct-untrained-smiles" => "MoLFormer, Untrained",
        "ibm-research/MoLFormer-XL-both-10pct-smiles-kekule" => "MoLFormer, Kekule",
        "ibm-research/MoLFormer-XL-both-10pct-smiles-canonical" => "MoLFormer, Canonical",
        "seyonec/ChemBERTa-zinc-base-v1-untrained-smiles" => "ChemBERTa, Untrained",
        "seyonec/ChemBERTa-zinc-base-v1-smiles-kekule" => "ChemBERTa, Kekule",
        "seyonec/ChemBERTa-zinc-base-v1-smiles-canonical" => "ChemBERTa, Canonical",
    ]

    # Effect Model
    o = load_scores(model_dirs, heuristics_dir, baseline_dirs)
    for (_, dataset) in pairs(o)
        required = first.(vcat(main_models, all_models))
        absent = setdiff(required, names(dataset.score))
        isempty(absent) || error("Missing molecular-surprise comparisons: $absent")
    end
    mkpath(output_dir)
    m_crowd = model_auroc(o.crowd.auroc; base_model="mist/$(first(bases))")
    m_ba = model_auroc(o.ba.auroc; base_model="mist/$(first(bases))")
    regtable(m_crowd, m_ba;
        render = LatexTable(),
        file=joinpath(output_dir, "interp_surprise.tex"),
        regression_statistics=[
             RegressionTables.Nobs,
             RegressionTables.DOF,
             RegressionTables.R2,
             (m -> mean(abs, residuals(m))) => "MAE",
             (m -> sqrt(mean(abs2, residuals(m)))) => "RMSE",
        ]
    )

    size_large = (4.5inch, 2.3inch)

    with_theme(MISTStyle.theme()) do
        datasets = [
            ("crowd", o.crowd, ["meanComplexity" => "Chemist"]),
            ("ba", o.ba, [])
        ]

        correlations = [
            "spearman" => corspearman,
            "pearson" => cor,
            "cosine" => cosine_similarity,
        ]
        model_sets = [
            "all" => vcat(main_models, all_models),
            missing => main_models,
        ]

        for ((set_name, models), (ds_name, ds), (cor_name, correlation)) in Iterators.product(model_sets, datasets, correlations)
            filename = "interp_surprise_" * join(skipmissing([set_name, ds_name, cor_name]), "_")
            kwargs = (; correlation)
            if !ismissing(set_name) && set_name == "all"
                kwargs = (; kwargs..., size=size_large)
            end
            if ds_name == "crowd"
                models = vcat(["meanComplexity" => "Chemist"], models)
            end
            fig = figure_interp_surprise(ds.score, ds.auroc, models; kwargs...)
            MISTStyle.savefig(filename, fig; fig_dir=output_dir)
        end

        for (name, df, baseline, cmax) in [
            ("model_score_grid_crowd", o.crowd.score, "meanComplexity", 100),
            ("model_score_grid_crowd_ma", o.crowd.score, "assembly-index", 100),
            ("model_score_grid_ba", o.ba.score, "SAScore", 500),
            ("model_score_grid_ba_ma", o.ba.score, "assembly-index", 500),
        ]
            MISTStyle.savefig(name, model_score_grid(df, baseline; cmax); fig_dir=output_dir)
        end
    end
end

function (@main)(args)
    option(name) = args[findfirst(==(name), args) + 1]
    options(name) = [args[i + 1] for i in eachindex(args) if args[i] == name]
    bases = options("--base")
    baseline_dirs = options("--baseline-dir")
    output = option("--output-dir")
    create_figures(bases, options("--model-dir"), option("--heuristics-dir"), output; baseline_dirs)
    open(joinpath(output, "figures.json"), "w") do io
        JSON.print(io, Dict("models" => bases, "figures" => filter(f -> endswith(f, ".pdf"), readdir(output))))
    end
end
