using TokenizerStats: TokenizerStats, tokenizer_label, moments!, hist_nbins, find, tokenusage!, classify_tokenizer, powerlaw!
using PythonCall
using GLMakie
using CairoMakie
using LinearAlgebra: normalize
using StatsBase: StatsBase, Histogram, fit, AbstractWeights, mean, Weights, median, mean_and_std, quantile
using JSON
using BSON: BSON
using JLD2: jldopen
using DataFrames
using CategoricalArrays: categorical, levelcode, levels
using OnlineStats: OrderedDict
using Colors: distinguishable_colors, weighted_color_mean, RGBA
using GLM
using CSV: CSV
using Format

GLMakie.activate!()

include("summarize.jl")

const GIT_ROOT = strip(read(`git rev-parse --show-toplevel`, String))

const CLASS_MARKER = Dict(
    :atomic => :x,
    :nlp => :diamond,
    :nlp_based => :cross,
    :ours => :star5,
)

# Figure Units
const pt = 3 / 4
const inch = 96

plt_tokenizers = [
    "smirk",
    "smirk-gpe-50k-nmb-ss",
    "character",
    "ibm/MoLFormer-XL-both-10pct-oov",
    "devalab/molgpt-moses",
    "devalab/molgpt-guacamol",
    "rxn4chemistry/rxnfp",
    "sagawa/ReactionT5-product-prediction",
    "sagawa/ReactionT5-yield-prediction",
    "seyonec/ChemBERTa-zinc-base-v1",
    "SmilesPE/SPE_ChEMBL",
    "ChangwenXu98/TransPolymer",
    "google/gemma-7b",
    "meta-llama/Meta-Llama-3.1-8B",
    "Xenova/gpt-4o",
]


function theme()
    Theme(
        rowgap=2,
        colgap=2,
        fonts=(;
            :regular => TokenizerStats.findfont("Helvetica", "Regular"),
            :bold => TokenizerStats.findfont("Helvetica", "Bold"),
        ),
        fontsize=8pt,
        size=(246, 152),
        figure_padding=(1, 15, 2, 2),
        CairoMakie=(;
            pt_per_unit=2,
            px_per_unit=300 / inch
        ),
        GLMakie=(; px_per_unit=4, scalefactor=4, focus_on_show=false),
        palette=(;
            color=cgrad(:seaborn_muted, 10),
            linestyle=[:solid, :dot, :dashdot],
        ),
        Lines=(;
            cycle=Cycle([:color, :linestyle], covary=true),
        ),
        Axis=(;
            spinewidth=0.5,
            ylabelpadding=3pt,
            yticksize=3,
            ytickwidth=0.5,
            yminortickwidth=0.5,
            yminorticksize=2,
            xtickwidth=0.5,
            xticksize=3,
            xminortickwidth=0.5,
            xminorticksize=2,
            xgridwidth=0.5,
            ygridwidth=0.5,
            xminorgridwidth=0.5,
            yminorgridwidth=0.5,
        ),
        Legend=(;
            titlegap=0,
            patchsize=(8, 8),
            rowgap=2pt,
            colgap=8,
            groupgap=4pt,
            framewidth=0.5,
            tellheight=false,
            tellwidth=false,
            padding=(2pt, 2pt, 2pt, 2pt),
        ),
        Colorbar=(;
            spinewidth=0.5,
            tickwidth=0.5,
            ticksize=2,
        ),
        Scatter=(;
            markersize=8pt,
            marker=:x,
        ),
    )
end


""" Save duplicate figures for publication and web """
function savefig(name::String, f::Figure; dpi=300)
    save(joinpath(@__DIR__, "fig", name * ".pdf"), f; pt_per_unit=1)
    save(joinpath(@__DIR__, "fig", name * ".png"), f; px_per_unit=dpi / inch)
    return nothing
end

function all_figures()
    mkpath(joinpath(@__DIR__, "fig"))
    CairoMakie.activate!()

    loss_stats = model_loss_stats()
    info_loss = info_loss_stats()
    molnet_info_loss = TokenizerStats.avg_molnet_info_loss(info_loss)
    usage_stats = TokenizerStats.usage_stats()
    dfp, dff, dft = filter_runs(transformer_models()...)

    with_theme(theme()) do
        # Token Usage
        savefig("token_usage", figure_token_usage())
        savefig("fertility", figure_fertility(usage_stats, loss_stats, dfp))
        savefig("oov_rate", figure_oov_rate())
        # savefig("jaccard", figure_jaccard())

        # Transformers
        savefig("ngram_vs_transformer", figure_ngram_vs_transformer(loss_stats, dfp, dff))
        f, df = figure_tf_finetune(dff, dft)
        savefig("tf_finetune", f)
        CSV.write(joinpath("stats/tf_model_summary.csv"), df)

        # N-Gram Analysis
        savefig("ngram_fits", figure_ngram_fits(loss_stats))
        # savefig("ngram_unk_log_odds", figure_ngram_info_loss())
        savefig("kl_v_info_loss", figure_kl_v_info_loss(loss_stats, molnet_info_loss))
        # savefig("info_loss_ref_tokenzier", figure_info_loss_ref_tokenizer(info_loss))

        # savefig("ngrma_unk_log_odds_cobalt", figure_ngram_info_loss(;
        #     smi="[Cl-][Co+2@OH1]([Cl-])([NH3])([NH3])([NH3])[NH3]",
        #     token_colors=("[NH3]" => :magenta, "[Co+2@OH1]" => :turquoise, "[Cl-]" => :orange)
        # ))

        # Example n-gram predictions
        compunds = [
            "caffine" => "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "lsd" => "CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C",
            "cortisol" => "O=C4\\C=C2/[C@]([C@H]1[C@@H](O)C[C@@]3([C@@](O)(C(=O)CO)CC[C@H]3[C@@H]1CC2)C)(C)CC4",
        ]
        for (name, smi) in compunds
            for direction in [:forward, :bidirectional]
                savefig("log_prob_$(direction)_$name", figure_ngram_prediction(smi; direction))
            end
        end
    end
end


function fit_hist(counts; kwargs...)
    x = map(Base.Fix1(parse, Int) ∘ string, collect(keys(counts)))
    w = Int.(collect(values(counts))) |> StatsBase.FrequencyWeights
    nbins = hist_nbins(:sturges, x, w)
    h = fit(Histogram, x, w; nbins, kwargs...)
    h = normalize(h)
    centers = StatsBase.midpoints(first(h.edges))
    return centers, h.weights
end

function collect_results(glob=r"stats-.*\.json")
    tokenizers = Dict{String,String}()
    for result in find(joinpath(@__DIR__, "stats"), glob)
        name = joinpath(splitpath(relpath(result, @__DIR__))[2:end-1])
        tokenizers[name] = result
    end
    return tokenizers
end

function figure_token_usage(stats_dir=joinpath(@__DIR__, "stats"))
    f = Figure(;
        size=(6inch, 4.5 * inch),
        figure_padding=(1, 1, 1, 5),
    )
    ax = Axis(f[1, 1];
        limits=((1, 2500), (0, 25)),
        xlabel="Token Rank",
        ylabel="Information Content [nats]",
        xscale=log10,
        xminorticksvisible=true,
        xminorticks=IntervalsBetween(5),
        xminorgridvisible=true,
    )

    rows = []
    tokenizers = tokenizers_info()
    pop!(tokenizers, "smirk-gpe-50k-nmb-ss", nothing)
    for name_or_path in keys(tokenizers)
        tok_info = tokenizers[name_or_path]
        usage_file = joinpath(stats_dir, tok_info["name_or_path"], "realspace", "usage.jld2")
        isfile(usage_file) || continue

        stats = jldopen(usage_file)
        usage = stats["train"][:ngrams][1]
        usage = Dict{Int,Int}(only(k) => v for (k, v) in pairs(usage))
        vocab_size = stats["tokenizer"][:vocab_size]
        unk_count = pop!(usage, stats["tokenizer"][:unk_token_id], 0)
        close(stats)

        c_token = TokenizerStats.collate_token_usage(usage, vocab_size; smoothing=0)
        p_token = c_token ./ sum(c_token)

        sort!(p_token, rev=true)
        efficiency = sum(p -> -p * log(p) / log(vocab_size), filter(>(0), p_token))
        H = sum(p -> p > 0 ? -p * log(p) : 0, p_token; init=0.0)
        p_unk = unk_count / sum(c_token)


        class = tok_info["tokenizer_class"]
        encoding = tok_info["encoding"]
        class_label = class * (encoding == "smiles" ? "" : ", $(encoding)")
        class_label = replace(class_label, "smirk" => "ours")
        push!(rows, (;
            name_or_path,
            name=tok_info["name"],
            usage,
            H,
            efficiency,
            vocab_size,
            class,
            class_label,
            p_unk
        ))
    end
    df = DataFrame(rows)
    sort!(df, [order(:efficiency, rev=true), :name, :vocab_size])
    display(select(df, Not(:usage)))

    linestyles = [:dot, :dashdot, :dashdotdot, :solid]
    colors = distinguishable_colors(nrow(df), [colorant"white", colorant"black"];
        dropseed=true,
        lchoices=range(0, stop=70, length=15), # Avoid light colors
    )

    for (idx, row) in enumerate(eachrow(df))
        tokenusage!(ax, row.usage, row.vocab_size;
            smoothing=0,
            label=format("{:.1%} - {:s} ({:s})", row.efficiency, row.name, row.class_label),
            linestyle=linestyles[idx%length(linestyles)+1],
            linewidth=1,
            color=colors[idx],
        )
    end


    Legend(f[1, 2], ax; tellheight=true, tellwidth=true, patchsize=(12, 6), valign=:top)
    colgap!(f.layout, 2)
    resize_to_layout!(f)
    return f
end


function figure_vocab_entropy(results::Dict)
    f = Figure(size=(600, 300))
    ax = Axis(f[1, 1];
        ylabel="Entropy [bits]",
        limits=(nothing, (-2.5, 25)),
        xticklabelrotation=0.4,
    )
    names = String[]
    categories = Int[]
    values = Float64[]
    weights = Float64[]
    for (idx, (name, file)) in enumerate(pairs(results))
        stats = JSON.parsefile(file)
        push!(names, name)
        hist = stats["entropy"]["hist"]
        append!(categories, repeat([idx], length(hist["centers"])))
        append!(values, float.(hist["centers"]))
        append!(weights, hist["counts"] ./ sum(hist["counts"]))
    end
    violin!(ax, categories, values; weights=weights, show_median=true)
    ax.xticks[] = (1:length(names), names)

    return f
end

function figure_fertility(usage_stats=nothing, loss_stats=nothing, tf_stats=nothing)
    f = Figure(size=(2.75inch, 2.25inch))
    ax = Axis(f[1, 1];
        limits=((0, 75), nothing),
        ylabel="N-Gram\nVal. Loss [nats/token]",
        xlabel="Tokenizer Fertility",
    )
    if isnothing(usage_stats)
        usage_stats = TokenizerStats.usage_stats()
    end
    if isnothing(loss_stats)
        loss_stats = TokenizerStats.model_loss_stats()
    end
    if isnothing(tf_stats)
        tf_stats, _ = transformer_models()
    end
    tokenizers = tokenizers_info()

    # Select the best n-gram model for each tokenizer
    val_loss = subset(loss_stats,
        :split => ByRow(==("val")),
        :dataset => ByRow(==("realspace")),
        :tokenizer => ByRow(x -> haskey(tokenizers, x)),
    )
    best_models = combine(groupby(val_loss, :tokenizer)) do gdf
        sort!(gdf, :avg_model_loss; rev=false)
        return gdf[1, :]
    end

    fertility = subset(usage_stats,
        :split => ByRow(==("train")),
        :dataset => ByRow(==("realspace")),
    )
    select!(fertility, [:tokenizer, :avg_fertility, :std_fertility])

    df = leftjoin(best_models, fertility, on=[:tokenizer])
    df.class = map(name -> tokenizers[name]["tokenizer_class"], df.tokenizer)
    df = leftjoin(df, select(tf_stats, :tokenizer, :val_loss => :pretrained_val_loss), on=:tokenizer)

    sort!(df, [:class, :avg_model_loss])
    dropmissing!(df)

    # Don't fit on outlier 
    dff = subset(df, :tokenizer => ByRow(!=("ncfrey/ChemGPT-4.7M")))

    # Regression model
    ols = lm(@formula(avg_model_token_loss ~ avg_fertility + 1), dff)
    display(ols)

    # Plot results
    ablines!(ax, coef(ols)[1], coef(ols)[2]; label="Linear Fit", color=:red, linewidth=0.5)
    df.class = categorical(string.(df.class))
    for gdf in groupby(df, :class)
        class = first(gdf.class)
        scatter!(ax, gdf.avg_fertility, gdf.avg_model_token_loss;
            color=levelcode(class),
            colorrange=(1, length(levels(df.class))),
            label=string(class),
            colormap=:Set2_8,
        )
    end

    # Repeat with transformer models
    ols_tf = lm(@formula(pretrained_val_loss ~ avg_fertility + 1), dff)
    display(ols_tf)

    ax_tf = Axis(f[2, 1]; xlabel="Fertility", ylabel="Transformer\nVal. Loss [nats/token]")
    ablines!(ax_tf, coef(ols_tf)[1], coef(ols_tf)[2]; label="Linear Fit", color=:red, linewidth=0.5)
    for gdf in groupby(df, :class)
        class = first(gdf.class)
        scatter!(ax_tf, gdf.avg_fertility, gdf.pretrained_val_loss;
            color=levelcode(class),
            colorrange=(1, length(levels(df.class))),
            label=string(class),
            colormap=:Set2_8,
        )
    end

    linkxaxes!(ax_tf, ax)
    hidexdecorations!(ax; grid=false)
    axislegend(ax, position=:lb, fontsize=8)
    resize_to_layout!(f)

    return f
end

function collate_atomic_oov(key::String, results::Dict)
    !haskey(results, key) && return missing, missing
    r = results[key]
    total = r["nobs"] + r["failed_encode"]
    covered = total - r["oov"] - r["failed_encode"]
    return (covered / total), (r["failed_encode"] / total)
end

function collate_atomic_oov(key::Regex, results::Dict)
    nobs = 0
    oov = 0
    failed_encode = 0
    valid = false
    for (k, v) in results
        if !isnothing(match(key, k))
            valid = true
            nobs += v["nobs"] + v["failed_encode"]
            oov += v["oov"]
            failed_encode += v["failed_encode"]
        end
    end
    valid || return missing, missing
    covered = nobs - oov - failed_encode
    return (covered / nobs), (failed_encode / nobs)
end

function figure_oov_rate(stats_dir=joinpath(@__DIR__, "stats"))
    datasets = OrderedDict(
        "Elements" => "elements",
        "Bonds" => "bonds",
        "Isotopes" => "isotopes",
        "Carbon Rings" => "rings",
        "Ions" => "charged_elements",
        "Chirality" => "chiral_elements",
        "Charged, Chiral Isotopes" => "charged_chiral_isotopes",
        "MoleculeNet" => r"^MoleculeNet/",
        "tmQM" => "tmQM",
    )

    plt_tokenizers = [
        "smirk",
        "smirk-gpe-50k-mb-ss",
        "ibm/MoLFormer-XL-both-10pct-oov",
        "ibm/materials.smi-ted-light",
        "seyonec/ChemBERTa-zinc-base-v1",
        "lbnlp/MatBERT-uncased",
        "devalab/molgpt-guacamol",
        "rxn4chemistry/rxnfp",
        "MolecularAI/Chemformer",
        "sagawa/ReactionT5-yield-prediction",
        "SmilesPE/SPE_ChEMBL",
        "mikemayuare/SMILYAPE",
        "mikemayuare/SELFYAPE",
        "HUBioDataLab/SELFormer",
        "ibm/materials.selfies-ted",
        "ChangwenXu98/TransPolymer",
        "ncfrey/ChemGPT-4.7M",
    ]

    tokenizers = tokenizers_info()
    tok_names = String[]
    tok_id = Int[]
    oov_rate = Float64[]
    ds_grp = Int[]
    enc_grp = Int[]
    for (idx, name_or_path) in enumerate(plt_tokenizers)
        tok_info = tokenizers[name_or_path]
        push!(tok_names, tok_info["name"])
        oov_stat_file = joinpath(stats_dir, tok_info["name_or_path"], "oov.json")
        oov_stats = isfile(oov_stat_file) ? JSON.parsefile(oov_stat_file) : Dict()

        # Collate group oov rates
        tok_info["oov_rate"] = Dict{String,Float64}()
        for (gdx, group_key) in enumerate(collect(values(datasets)))
            group_oov_rate, group_encode_rate = collate_atomic_oov(group_key, oov_stats)
            append!(tok_id, (idx, idx))
            append!(oov_rate, (group_oov_rate, group_encode_rate))
            append!(ds_grp, (gdx, gdx))
            append!(enc_grp, (0, 1))
        end
    end

    # Darken encoding failures
    colormap = cgrad(:Set1_9, length(datasets); categorical=true)
    colormap = [c for c in colormap]
    append!(colormap, map(c -> weighted_color_mean(0.2, RGBA(colorant"black"), c), colormap))
    color = ds_grp .+ length(datasets) .* enc_grp

    # Create figure
    f = Figure(;
        size=(7inch, 2inch),
        figure_padding=(5, 1, 1, 5),
    )
    ax = Axis(f[1, 1];
        ylabel="Tokenizer Coverage",
        limits=((0, length(tok_names) + 1), (0, 1)),
        ytickformat="{:.0%}",
        xticklabelrotation=0.4,
        xticks=(1:length(tok_names), tok_names),
        xticklabelsize=10pt,
        yminorticks=IntervalsBetween(5),
        yminorticksvisible=true,
        yminorgridvisible=true,
        xgridvisible=false,
    )
    h = barplot!(ax, tok_id, oov_rate;
        dodge=ds_grp,
        color,
        stack=enc_grp,
        colorrange=(1, length(colormap)),
        colormap,
        strokewidth=0.1,
        strokecolor=:black,
    )
    ds_elements = map(1:length(datasets)) do gdx
        PolyElement(polycolor=gdx, colormap=h.colormap, colorrange=h.colorrange)
    end
    Legend(f[1, 2], ds_elements, collect(keys(datasets));
        tellheight=false, tellwidth=true, orientation=:vertical,
        patchstrokecolor=:black,
        framevisible=false,
    )

    resize_to_layout!(f)
    return f
end


function figure_ngram_fits(df=nothing; colormap=:Set2_5)
    if isnothing(df)
        df = TokenizerStats.model_loss_stats()
    end
    plt_toks = [
        "smirk",
        "smirk-gpe-50k-nmb-ss",
        "ibm/MoLFormer-XL-both-10pct-oov",
        "ibm/materials.smi-ted-light",
        "devalab/molgpt-moses",
        "rxn4chemistry/rxnfp",
        "SmilesPE/SPE_ChEMBL",
        "MolecularAI/Chemformer",
        "mikemayuare/SMILYAPE",
        "mikemayuare/SELFYAPE",
        "HUBioDataLab/SELFormer",
        "seyonec/ChemBERTa-zinc-base-v1",
        "ncfrey/ChemGPT-4.7M",
        "ChangwenXu98/TransPolymer",
        "sagawa/ReactionT5-product-prediction",
        "facebook/galactica-6.7b",
        "meta-llama/Llama-3.2-1B",
        "Xenova/gpt-4o",
        "google/gemma-7b",
    ]
    df = subset(df,
        :tokenizer => ByRow(x -> x in plt_toks),
        :split => ByRow(==("val")),
    )
    df.tokenizer = categorical(df.tokenizer; levels=plt_toks)
    tokenizers = tokenizers_info()

    # Set up figure
    f = Figure(; size=(7inch, 3inch))
    tokenizer = levels(df.tokenizer)
    plt_label = map(tokenizer) do tok
        name = tokenizers[tok]["name"]
        class = tokenizers[tok]["tokenizer_class"]
        return "$name - $class"
    end
    ax_kwargs = (;
        limits=(nothing, (0, nothing)),
        xticks=(1:length(tokenizer), plt_label),
        xticklabelrotation=0.4,
        xticksvisible=false,
        xgridvisible=false,
    )
    ax_pretrain = Axis(f[1, 1];
        ylabel="Enimine REAL Space\nCross Entropy Loss [nats/token]",
        ax_kwargs...
    )
    hidexdecorations!(ax_pretrain)
    ax_molnet = Axis(f[2, 1];
        ylabel="MoleculeNet\nCross Entropy Loss [nats/token]",
        ax_kwargs...
    )
    linkxaxes!(ax_pretrain, ax_molnet)
    ax = Axis(f[1:2, 2];
        ylabel="Molecule Net [nats/token]",
        xlabel="Enimine REAL Space\n[nats/token]",
        limits=((0, nothing), (0, nothing)),
    )
    colsize!(f.layout, 2, Aspect(1, 1))
    colgap!(f.layout, 5)

    # N-Gram Legend
    ds_elements = map(1:5) do gdx
        PolyElement(polycolor=gdx;
            colormap,
            colorrange=(1, 5),
        )
    end
    Legend(f[1, 1], ds_elements, ["unigram", "bigram", "trigram", "4-gram", "5-gram"];
        tellheight=false, tellwidth=false, orientation=:horizontal,
        framevisible=true,
        patchstrokecolor=:black,
        patchstrokewidth=1,
        margin=(2, 2, 2, 2),
        padding=3,
        valign=:top,
        halign=:left,
    )

    # Pretraining
    df_pretrain = subset(df,
        :dataset => ByRow(==("realspace")),
    )
    barplot!(ax_pretrain, levelcode.(df_pretrain.tokenizer), df_pretrain.avg_model_token_loss;
        dodge=df_pretrain.ngram,
        color=df_pretrain.ngram,
        colormap,
    )

    # Finetune
    df_finetune = subset(df,
        :dataset => ByRow(!=("realspace")),
    )
    df_finetune = combine(groupby(df_finetune, [:tokenizer, :ngram])) do gdf
        return (;
            avg_model_loss=mean(gdf.avg_model_token_loss, Weights(gdf.samples)),
        )
    end
    barplot!(ax_molnet, levelcode.(df_finetune.tokenizer), df_finetune.avg_model_loss;
        dodge=df_finetune.ngram,
        color=df_finetune.ngram,
        colormap,
    )

    # Scatter Plot of Pretrain vs. Finetune 
    df = leftjoin(
        select(subset(df_pretrain, :ngram => ByRow(==(5))), :tokenizer, :avg_model_token_loss => :pretrain),
        select(subset(df_finetune, :ngram => ByRow(==(5))), :tokenizer, :avg_model_loss => :finetune);
        on=:tokenizer,
    )
    df.class = map(name -> tokenizers[name]["tokenizer_class"], df.tokenizer) |> categorical
    df.domain = map(name -> tokenizers[name]["domain"], df.tokenizer) |> categorical
    markers = [:x, :+, :diamond, :square]
    h = scatter!(ax, df.pretrain, df.finetune;
        marker=map(i -> markers[levelcode(i)], df.domain),
        colormap=:Dark2_6,
        color=levelcode.(df.class),
        colorrange=(1, length(levels(df.class))),
    )
    domains = map(enumerate(levels(df.domain))) do (i, domain)
        MarkerElement(label=string(domain), marker=markers[i], color=:black)
    end
    classes = map(enumerate(levels(df.class))) do (i, class)
        PolyElement(
            label=string(class),
            marker=:x,
            color=i,
            colormap=h.colormap,
            colorrange=h.colorrange
        )
    end
    labels(x) = map(e -> e.label[], x)
    Legend(f[2, 1:2],
        [domains, classes],
        [labels(domains), labels(classes)],
        ["Domain", "Tokenizer Class"];
        halign=:right, valign=:bottom,
        nbanks=2,
        titleposition=:top,
        margin=(5, 5, 5, 5),
    )




    rowgap!(f.layout, 2)
    resize_to_layout!(f)
    return f
end

function figure_ngram_prediction(smi; direction=:forward)
    f = Figure(size=(3.42inch, 3inch))
    cb = Colorbar(f[1:3, 4];
        label="Log Probability",
        colormap=:lipari,
        colorrange=(-10, 0),
        tickformat="{:2d}",
    )

    tokenizers = tokenizers_info()
    path(name_or_path) = (joinpath(@__DIR__, "stats", name_or_path, "realspace", "usage.jld2"), tokenizers[name_or_path]["name"])
    tok_log_prob!(f[1, 1], cb, path("smirk")..., smi; direction)
    tok_log_prob!(f[1, 2], cb, path("ibm/MoLFormer-XL-both-10pct-oov")..., smi; direction)
    tok_log_prob!(f[1, 3], cb, path("seyonec/ChemBERTa-zinc-base-v1")..., smi; direction)
    tok_log_prob!(f[2, 1], cb, path("devalab/molgpt-moses")..., smi; direction)
    tok_log_prob!(f[2, 2], cb, path("rxn4chemistry/rxnfp")..., smi; direction)
    tok_log_prob!(f[2, 3], cb, path("MolecularAI/Chemformer")..., smi; direction)
    tok_log_prob!(f[3, 1], cb, path("meta-llama/Meta-Llama-3.1-8B")..., smi; direction)
    tok_log_prob!(f[3, 2], cb, path("Xenova/gpt-4o")..., smi; direction)
    tok_log_prob!(f[3, 3], cb, path("google/gemma-7b")..., smi; direction)

    # Format plot
    Label(f[:, 0], smi, rotation=pi / 2, fontsize=length(smi) > 40 ? 6 : 8, padding=(0, 2, 0, 0))
    Label(f[end+1, :], "Predicted Tokens", fontsize=8)
    resize_to_layout!(f)
    rowgap!(f.layout, 2)
    colgap!(f.layout, 1)

    return f
end

function tok_log_prob!(f, cb, file, name, smi, max_vocab=50; direction=:forward)
    ngram, tok, info = TokenizerStats.load_ngram_model(file)
    code = pyconvert(Vector{Int}, tok(smi)["input_ids"])

    if direction == :forward
        P = TokenizerStats.autoregressive_log_prob(ngram, code)
    elseif direction == :bidirectional
        P = TokenizerStats.fb_log_probability(ngram, code)
    else
        error("unknown type $type")
    end
    l = TokenizerStats.cross_entropy(P, code)

    vocab = TokenizerStats.nonspecial_vocab(ngram)
    P = P[vocab.+1, :]

    # Truncate Vocab
    max_vocab = min(max_vocab, size(P, 1))
    name = size(P, 1) > max_vocab ? "*" * name : name
    sdx = sortperm(vec(sum(P; dims=2)); rev=true)
    P = P[sdx[1:max_vocab], :]

    ax = Axis(f;
        title="$(name): $(round(l; sigdigits=2))",
        limits=((0, size(P, 1)), (0, size(P, 2))),
        titlegap=1,
        xticksvisible=false,
        xticklabelsvisible=false,
        yticksvisible=false,
        yticklabelsvisible=false,
        spinewidth=0.5,
        aspect=1,
    )
    image!(ax, P; colormap=cb.colormap, colorrange=cb.colorrange, interpolate=false)

    return nothing
end

function figure_ngram_info_loss(;
    smi="C(=Cc1ccccc1)C1=[O+][Cu-3]2([O+]=C(C=Cc3ccccc3)CC(c3ccccc3)=[O+]2)[O+]=C(c2ccccc2)C1",
    token_colors=("[O+]" => :turquoise, "[Cu-3]" => :magenta)
)
    f = Figure(size=72 .* (4.5, 1.7),
        figure_padding=(1, 1, 5, 1),
    )
    cb = Colorbar(f[1, 4];
        label="Log Odds Ratio",
        colormap=:vik,
        colorrange=(-50, 50),
        tellheight=true,
    )

    rsmi, token_color = rich_smi(smi, token_colors...)

    # Load model
    ref_file = joinpath(@__DIR__, "stats", "character", "realspace/usage.jld2")
    ngram, ref_tok, ref_info = TokenizerStats.load_ngram_model(ref_file)
    ref_code = pyconvert(Vector{Int}, ref_tok(smi)["input_ids"])
    kwargs = (; ngram, ref_tok, ref_code, token_color)

    tok_info_loss!(f[1, 1], cb, "smirk", smi; kwargs...)
    tok_info_loss!(f[1, 2], cb, "ibm/MoLFormer-XL-both-10pct-oov", smi; kwargs...)
    tok_info_loss!(f[1, 2], cb, "SmilesPE/SPE_ChEMBL", smi; kwargs...)
    tok_info_loss!(f[2, 1], cb, "MolecularAI/Chemformer", smi; kwargs...)
    tok_info_loss!(f[1, 3], cb, "devalab/molgpt-moses", smi; kwargs...)
    tok_info_loss!(f[2, 3], cb, "rxn4chemistry/rxn_yields", smi; kwargs...)

    # Show vocab
    vocab = TokenizerStats.nonspecial_vocab(ngram)
    unk = pyconvert(Int, ref_tok.unk_token_id)
    filter!(!=(unk), vocab)
    rvocab = Makie.RichText[]
    for (i, id) in enumerate(vocab)
        token = pyconvert(String, ref_tok.decode(id))
        pad = i % 5 == 0 ? "  " : ""
        push!(rvocab, rich(token * pad))
    end

    # Format plot
    Label(f[0, :], rsmi, fontsize=6)
    # Label(f[end+1, :], rich(rvocab...), fontsize=8)
    resize_to_layout!(f)
    colgap!(f.layout, 3)
    rowgap!(f.layout, 3)

    return f
end

function rich_smi(smi::String, colors::Pair...; colormap=:viridis)
    out = []
    colors = Dict(colors)
    idx = firstindex(smi)
    matches = findall(r"\[.*?\]", smi)
    cmap = cgrad(colormap, length(matches))
    token_color = Vector(undef, length(smi))
    token_color .= :black
    for m in matches
        if idx != prevind(smi, first(m))
            r = idx:prevind(smi, first(m))
            token_color[r] .= :black
            push!(out, rich(smi[r]))
        end

        # Get color for segment
        segment = get(colors, smi[m], cmap[length(colors)+1])
        colors[smi[m]] = segment
        token_color[m] .= segment
        push!(out, rich(smi[m]; color=segment, font=:bold))
        idx = nextind(smi, last(m))
    end

    if idx != lastindex(smi)
        push!(out, rich(smi[idx:end]))
        token_color[idx:end] .= :black
    end
    return rich(out...), token_color
end

function box_token!(ax, token_id::Int, code_pos::Int; offset=0.0, kwargs...)
    token_id += 1
    point = [
        (token_id, code_pos),
        (token_id, code_pos + 1),
        (token_id + 1, code_pos + 1),
        (token_id + 1, code_pos),
        (token_id, code_pos),
    ]
    point = map(x -> (x[1] - offset, x[2] - offset), point)
    lines!(ax, point; color=:red, linewidth=0.1, kwargs...)
end

function tok_info_loss!(f, cb, tok::String, smi::String; ngram, ref_tok, ref_code, token_color=missing)

    # Load model
    name = tokenizers_info()[tok]["name"]
    tok = TokenizerStats.load_tokenizer(tok)
    code = pyconvert(Vector{Int}, tok(smi)["input_ids"])

    # Align both tokenizations
    smi_tokens = pyconvert(Vector{String}, tok.tokenize(smi))
    ref_tokens = pyconvert(Vector{String}, ref_tok.tokenize(smi))
    A = TokenizerStats.align_unknown(ref_tokens, smi_tokens)

    # Check for bos/eos tokens
    if length(code) - length(smi_tokens) == 2
        code = code[2:end-1]
    end
    # @assert length(code) == length(smi_tokens)

    # Compute information_loss from unknown tokens
    masked = map(!, vec(any(A; dims=2)))
    @assert length(masked) == length(ref_code)
    i = TokenizerStats.information_loss(ngram, ref_code, masked; N=2)

    # Remove special tokens
    vocab = TokenizerStats.nonspecial_vocab(ngram)
    unk = pyconvert(Int, ref_tok.unk_token_id)
    filter!(!=(unk), vocab)
    P = P[vocab.+1, :]
    Q = Q[vocab.+1, :]

    odds_ratio = @. (Q - log(1 - exp(Q))) - (P - log(1 - exp(P)))
    @info "masked for $name" masked_tokens = join(ref_tokens[masked], " ") extrema(odds_ratio)

    ax = Axis(f;
        title="$name: $(round(i; sigdigits=3))",
        xticks=collect(5:5:length(vocab)),
        xticksvisible=false,
        xticklabelsvisible=false,
        yticksvisible=false,
        yticklabelsvisible=false,
        aspect=1,
        spinewidth=0.5,)
    image!(ax, odds_ratio;
        colormap=cb.colormap,
        colorrange=cb.colorrange,
        highclip=cb.highclip,
        lowclip=cb.lowclip,
        interpolate=false,
    )

    # Highlight the correct token
    @info token_color
    for (code_pos, token_id) in enumerate(ref_code)
        color = ismissing(token_color) ? :black : token_color[code_pos]
        box_token!(ax, token_id, code_pos; linewidth=0.5, color)
    end

    return nothing
end

function figure_avg_info_loss()
    oov_stats = JSON.parsefile(joinpath(@__DIR__, "stats-atomic.json"))
end

struct Asinh
    a::Float64
end
Asinh() = Asinh(1)
(m::Asinh)(x::Real) = m.a * asinh(x / m.a)

Makie.inverse_transform(m::Asinh) = x -> m.a * sinh(x / m.a)
Makie.defined_interval(::Asinh) = Makie.defined_interval(identity)
Makie.defaultlimits(m::Asinh) = (0.0, 10 * m.a)

Makie.inverse_transform(::typeof(asinh)) = sinh
Makie.defined_interval(::typeof(asinh)) = Makie.defined_interval(identity)
Makie.defaultlimits(::typeof(asinh)) = (0.0, 10.0)

function figure_kl_v_info_loss(model_loss, info_loss; reference="character")
    tokenizers = tokenizers_info()

    model_loss = subset(model_loss,
        :dataset => ByRow(∉(["realspace", "tmqm"])),
        :split => ByRow(==("val")),
    )
    info_loss = subset(info_loss,
        :ref_tokenizer => ByRow(==(reference)),
    )
    model_loss = combine(groupby(model_loss, [:tokenizer, :split, :ngram])) do gdf
        @info gdf
        return (;
            avg_model_loss=mean(gdf.avg_model_loss, Weights(gdf.samples)),
            vocab_size=first(gdf.vocab_size),
            samples=sum(gdf.samples),
        )
    end
    select!(model_loss, Not([:samples]))
    select!(info_loss, Not([:samples, :vocab_size]))
    df = leftjoin(model_loss, info_loss, on=[:tokenizer, :ngram])
    plt_tokenizers = [
        "smirk-gpe-50k-nmb-ss" => :star8,
        "smirk" => :star5,
        "ibm/MoLFormer-XL-both-10pct-oov" => :diamond,
        "devalab/molgpt-moses" => :ltriangle,
        "devalab/molgpt-guacamol" => :rtriangle,
        # "rxn4chemistry/rxn_yields" => :cross,
        "rxn4chemistry/rxnfp" => :x,
        "sagawa/ReactionT5-product-prediction" => :circle,
        "ChangwenXu98/TransPolymer" => :dtriangle,
        "MolecularAI/Chemformer" => :utriangle,
        "SmilesPE/SPE_ChEMBL" => :hexagon,
    ]
    dropmissing!(df)
    subset!(df,
        :ngram => ByRow(>(1)),
        :split => ByRow(==("val")),
    )
    @info "Tokens with no info_loss" unique(df.tokenizer)
    subset!(df, :tokenizer => ByRow(x -> x in first.(plt_tokenizers)))
    df.tokenizer = categorical(df.tokenizer)
    display(df[!, [:tokenizer, :ngram, :avg_model_loss, :avg_info_loss]])

    f = Figure(; size=72 .* (4.5, 3), figure_padding=(1, 1, 1, 4))
    ax = Axis(f[1, 1];
        limits=(nothing, (-0.1, nothing)),
        xlabel="Cross Entropy Loss [nats]",
        ylabel="Information Loss [nats]",
        yticks=[0, 0.5, 2, 4, 16, 64, 256, 512],
        yscale=Asinh(0.2),
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(4),
        yminorgridvisible=true,
        xgridvisible=false,
    )
    h = scatter!(ax, df.avg_model_loss, df.avg_info_loss;
        colormap=:Set2_5,
        colorrange=(1, 5),
        color=df.ngram,
        marker=map(n -> Dict(plt_tokenizers)[n], df.tokenizer),
    )

    # Build legend
    ngram_elements = map(2:5) do gdx
        PolyElement(color=gdx, colorrange=h.colorrange, colormap=h.colormap)
    end
    ngram_labels = ["Bigram", "Trigram", "4-gram", "5-gram"]
    tokenizer_elements = map(levels(df.tokenizer)) do name
        MarkerElement(; marker=Dict(plt_tokenizers)[name], color=:black)
    end
    tokenizer_labels = map(levels(df.tokenizer)) do name_or_path
        return tokenizers[name_or_path]["name"]
    end
    Legend(f[1, 1],
        [ngram_elements, tokenizer_elements],
        [ngram_labels, tokenizer_labels],
        ["n-gram", "Tokenizer"];
        tellheight=false,
        tellwidth=false,
        halign=:right,
        valign=:top,
    )
    resize_to_layout!(f)

    return f
end

function figure_info_loss_ref_tokenizer(df=TokenizerStats.avg_molnet_info_loss())
    tokenizers = ("character", "meta-llama/Meta-Llama-3.1-8B")
    subset!(df,
        :split => ByRow(==(:val)),
        :ngram => ByRow(>(1)),
        :ref_tokenizer => ByRow(in(tokenizers)),
    )
    select!(df, [:tokenizer, :ref_tokenizer, :ngram, :avg_info_loss])
    df = unstack(df, [:tokenizer, :ngram], :ref_tokenizer, :avg_info_loss)
    dropmissing!(df)
    @info "Info Loss" characters = extrema(df.character) llama = extrema(df[!, "meta-llama/Meta-Llama-3.1-8B"])
    @info "Tokenizers" unique(df.tokenizer)
    display(subset(df, :ngram => ByRow(==(5))))

    f = Figure(; size=72 .* (6.5, 3))
    ax = Axis(f[1, 1];
        title="Information Loss from Unknown Tokens [nats]",
        limits=((0, 30), (0, 0.75)),
        xlabel=TOKENIZERS[tokenizers[1]],
        ylabel=TOKENIZERS[tokenizers[2]],
        aspect=1,
        xscale=sqrt,
        yscale=sqrt,
        xminorticksvisible=true,
        xminorticks=IntervalsBetween(10),
        xminorgridvisible=true,
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(10),
        yminorgridvisible=true,
    )
    lines!(ax, range(0, 30; length=100), range(0, 30; length=100); color=:black, linestyle=:dash)
    scatter!(ax, df[!, tokenizers[1]], df[!, tokenizers[2]];
        marker=:x,
        color=df.ngram,
    )

    model_loss = TokenizerStats.model_loss_stats()
    model_loss = subset(model_loss,
        :dataset => ByRow(!=("realspace_v4_dev")),
        :training_dataset => ByRow(==("realspace_v4_dev")),
        :split => ByRow(==(:val)),
        :tokenizer => ByRow(in(tokenizers)),
    )
    model_loss = combine(groupby(model_loss, [:tokenizer, :ngram])) do gdf
        return (;
            avg_model_loss=mean(gdf.avg_model_loss, Weights(gdf.samples)),
        )
    end
    model_loss = unstack(model_loss, :ngram, :tokenizer, :avg_model_loss)
    dropmissing!(model_loss)

    ax = Axis(f[1, 2];
        title="Cross-Entropy Loss [nats]",
        limits=((0, 225), (0, 225)),
        aspect=1,
        xlabel=TOKENIZERS[tokenizers[1]],
        ylabel=TOKENIZERS[tokenizers[2]],
    )
    hab = ablines!(ax, 0, 1; color=:black, linestyle=:dash, label="Parity")
    h = scatter!(ax, model_loss[!, tokenizers[1]], model_loss[!, tokenizers[2]];
        marker=:x,
        color=df.ngram,
        colorrange=(1, 5),
    )
    ngram_labels = ["Unigram", "Bigram", "Trigram", "4-gram", "5-gram"]
    ngram_elements = map(1:length(ngram_labels)) do gdx
        MarkerElement(; color=gdx, colorrange=h.colorrange, colormap=h.colormap, marker=h.marker)
    end
    Legend(f[1, 2], [hab, ngram_elements...], [hab.label, ngram_labels...];
        tellheight=false,
        tellwidth=false,
        halign=:right,
        valign=:bottom,
        margin=(10, 10, 5, 10),
    )
    Label(f[1, 1, TopLeft()], "a)";
        font=:bold,
        halign=:right,
        padding=(0, 15, 5, 0),
    )
    Label(f[1, 2, TopLeft()], "b)";
        font=:bold,
        halign=:right,
        padding=(0, 15, 5, 0),
    )

    resize_to_layout!(f)
    display(model_loss)

    return f
end

function figure_jaccard()
    tokenizers = collect(keys(TOKENIZERS))
    J, k = TokenizerStats.tokenizer_jaccard(tokenizers)

    # Reports stats
    @info "Jacard Index 90th percentile" quantile(filter(!=(1), vec(J)), 0.90)
    @info "Top 10 Jacard Index" sort(unique(vec(J)))[end-10:end]
    @info "Llama 3.1 v. GPT-4o" J[findfirst(==("meta-llama/Meta-Llama-3.1-8B"), k), findfirst(==("Xenova/gpt-4o"), k)]

    f = Figure(;
        size=72 .* (4.4, 3.5),
        figure_padding=(1, 1, 1, 5),
    )
    k = map(k -> TOKENIZERS[k], k)
    ax = Axis(f[1, 1];
        aspect=1,
        xticks=(1:length(k), k),
        yticks=(1:length(k), k),
        xticklabelrotation=pi / 4,
        xticklabelsize=6,
        yticklabelsize=6,
    )
    h = heatmap!(ax, J; colormap=:lajolla, colorrange=(0, 1))
    Colorbar(f[1, 2];
        tickformat="{:.0%}",
        label="Jaccard Index",
        colormap=h.colormap,
        colorrange=h.colorrange,
    )
    resize_to_layout!(f)
    return f
end


function pretraining_tokenizer_scheme(dfp, dff)

    dfi = TokenizerStats.info_loss_stats()
    tok_info_loss = combine(groupby(dfi, :tokenizer)) do gdf
        (; info_loss=mean(gdf.avg_info_loss, Weights(gdf.samples)))
    end
    dfp = leftjoin(dfp, tok_info_loss, on=:tokenizer)
    dff = leftjoin(dff, tok_info_loss, on=:tokenizer)

    lm(
        @formula(log10(val_loss) ~ encoding + tokenizer_class + info_loss), dfp;
        contrasts=Dict(
            :encoding => EffectsCoding(; base="smiles"),
            :tokenizer_class => EffectsCoding(; base="atomwise"),
        )
    ) |> display

    # Regression
    m_reg = lm(
        @formula(val_loss ~ tokenizer_class + dataset + encoding + info_loss),
        subset(dff, :task => ByRow(==("regression")));
        contrasts=Dict(
            :encoding => EffectsCoding(; base="smiles"),
            :token_class => EffectsCoding(; base="atomwise"),
            :dataset => EffectsCoding(),
        )
    )

    # Classification
    m_class = lm(
        @formula(val_loss ~ tokenizer_class + dataset + encoding + info_loss),
        subset(dff, :task => ByRow(==("binary")), :dataset => ByRow(!=("muv")));
        contrasts=Dict(
            :encoding => EffectsCoding(; base="smiles"),
            :token_class => EffectsCoding(; base="atomwise"),
            :dataset => EffectsCoding(),
        )
    )
    return m_class, m_reg
end

function figure_tf_finetune(dff, dft)
    f = Figure(; size=(5inch, 2.5inch), figure_padding=(1, 1, 1, 4))
    tokenizers = tokenizers_info()
    dff.tokenizer = categorical(dff.tokenizer)
    dff.dataset = categorical(dff.dataset)


    # Tabulated Results for SI
    colormap = distinguishable_colors(length(levels(dff.tokenizer)), [colorant"white", colorant"black"];
        dropseed=true,
        lchoices=range(0, stop=70, length=15), # Avoid light colors
    )

    # Get test results
    dft = subset(dft, :channel => ByRow(isnothing), :tok_group => ByRow(==("all")))
    select!(dft, Not(:tok_group, :channel))
    rename!(dft, :mean => :test_loss, :std => :test_loss_std, :id => :test_id)
    dff = leftjoin(dff, dft; on=[:id => :ckpt_id, :metric])

    # Merge smiles/canonical/kekule encoding into one bar (pick best by val loss)
    dff_all = dff # Save for reporting
    select!(dff, Not(:train_oov_loss, :val_oov_loss))
    dff = combine(groupby(dff, [:tokenizer, :dataset])) do gdf
        ids = argmin(gdf.val_loss)
        return gdf[ids, :]
    end

    # Plot Regression
    dfr = subset(dff, :task => ByRow(==("regression")))
    dfr.dataset = categorical(string.(dfr.dataset))
    ax = Axis(f[1, 1];
        limits=(nothing, (0, 1)),
        ylabel="Test R2",
        xticks=categorical_ticks(dfr.dataset),
        xticklabelrotation=0.4,
        yticks=LinearTicks(5),
        ytickformat="{:.0%}",
    )
    h = _finetune_results!(ax, dfr.dataset, dfr.test_loss, dfr.tokenizer;
        std=dfr.test_loss_std,
        colormap,
        colorrange=(1, length(colormap)),
    )

    # Skip MUV as it use AU-PRC, not AUROC
    dfc = subset(dff, :task => ByRow(==("binary")), :dataset => ByRow(!=("muv")))
    dfc.dataset = categorical(string.(dfc.dataset))
    ax = Axis(f[2, 1];
        limits=(nothing, (0, 1)),
        ylabel="Test AUROC",
        xticks=categorical_ticks(dfc.dataset),
        xticklabelrotation=0.4,
        yticks=LinearTicks(5),
        ytickformat="{:.0%}",
    )
    _finetune_results!(ax, dfc.dataset, dfc.test_loss, dfc.tokenizer;
        std=dfc.test_loss_std,
        colormap=h.colormap,
        colorrange=h.colorrange
    )

    # Dataset Legend
    ds_levels = levels(dff.tokenizer)
    ds_elements = map(1:length(ds_levels)) do gdx
        PolyElement(polycolor=gdx, colormap=h.colormap, colorrange=h.colorrange)
    end
    Legend(f[1:2, 2], ds_elements, map(tok -> tokenizers[tok]["name"], ds_levels);
        tellheight=false, tellwidth=true, orientation=:vertical,
        patchstrokecolor=:black,
        framevisible=false,
    )

    resize_to_layout!(f)
    return f, dff_all
end

function _finetune_results!(ax, x, y, dodge; std=nothing, colormap, colorrange=nothing)
    if isnothing(colorrange)
        colorrange = extrema(levelcode.(dodge))
    end

    h = barplot!(ax, levelcode.(x), y;
        dodge=levelcode.(dodge),
        colormap,
        colorrange,
        color=levelcode.(dodge),
    )

    if !isnothing(std)
        TokenizerStats.dodgederrorbars!(ax, levelcode.(x), y, std;
            dodge=h.dodge,
            width=h.width,
            n_dodge=h.n_dodge,
            gap=h.gap,
            dodge_gap=h.dodge_gap,
            linewidth=1,
            color=:black,
        )
    end

    return h
end

categorical_ticks(x) = (1:length(levels(x)), levels(x))

function figure_ngram_vs_transformer(loss_stats, dfp, dff)
    tokenizers = tokenizers_info()
    val_loss = subset(loss_stats,
        :split => ByRow(==("val")),
        :dataset => ByRow(==("realspace")),
        :tokenizer => ByRow(x -> haskey(tokenizers, x)),
    )
    best_models = combine(groupby(val_loss, :tokenizer)) do gdf
        sort!(gdf, :avg_model_loss; rev=false)
        return gdf[1, :]
    end
    select!(best_models, :tokenizer, :ngram, :avg_model_loss => :ngram_val_loss, :avg_model_token_loss => :ngram_val_token_loss)

    f = Figure(; size=(3.25inch, 1.5inch), figure_padding=(1, 1, 1, 4))

    # Get transformer model pretraining loss
    dfp = leftjoin(dfp, best_models, on=[:tokenizer])
    sort!(dfp, :val_loss)
    dfp.tokenizer = categorical(dfp.tokenizer, levels=unique(dfp.tokenizer))
    replace!(dfp.encoding,
        "smiles" => "SMILES",
        "smiles-canonical" => "Canonical SMILES",
        "selfies" => "SELFIES"
    )
    dfp.encoding = categorical(dfp.encoding, levels=["SMILES", "Canonical SMILES", "SELFIES"])

    toks = levels(dfp.tokenizer)
    ax = Axis(f[1, 1];
        limits=(nothing, (0, nothing)),
        ylabel="Transformer [nats/token]",
        xticklabelrotation=0.4,
        xticks=(1:length(toks), map(n -> tokenizers[n]["name"], toks)),
    )
    h = barplot!(ax, levelcode.(dfp.tokenizer), dfp.val_loss;
        dodge=replace(levelcode.(dfp.encoding), 3 => 2),
        color=levelcode.(dfp.encoding),
        colormap=:Set1_3,
        colorrange=(1, length(levels(dfp.encoding))),
    )

    ds_elements = map(enumerate(levels(dfp.encoding))) do (gdx, encoding)
        PolyElement(label=encoding, color=gdx, colorrange=h.colorrange, colormap=h.colormap)
    end
    Legend(f[1, 1], ds_elements, map(e -> e.label, ds_elements);
        tellheight=false, tellwidth=false, orientation=:vertical,
        framevisible=true,
        patchstrokecolor=:black,
        patchstrokewidth=1,
        margin=(2, 2, 2, 2),
        padding=3,
        valign=:top,
        halign=:left,
    )


    # Test if N-Gram's predict pretraining loss
    ols = glm(@formula(val_loss ~ 1 + log(ngram_val_token_loss)), dfp, Normal(), LogLink())
    display(ols)

    ax = Axis(f[1, 2];
        xlabel="n-gram [nats/token]",
        ylabel="Transformer [nats/token]",
        xscale=log10,
        yscale=log10,
        limits=((1, 3), (2e-2, 1)),
    )
    powerlaw!(ax, exp(coef(ols)[1]), coef(ols)[2]; linewidth=0.5, color=:black)

    scatter!(ax, dfp.ngram_val_token_loss, dfp.val_loss;
        color=levelcode.(dfp.encoding),
        colormap=h.colormap,
        colorrange=h.colorrange,
    )

    colsize!(f.layout, 1, Relative(2 / 3))
    colgap!(f.layout, 5)
    resize_to_layout!(f)

    return f
end
