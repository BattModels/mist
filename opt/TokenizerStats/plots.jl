using TokenizerStats: TokenizerStats, tokenizer_label, moments!, hist_nbins, find, tokenusage!, classify_tokenizer
using PythonCall
using GLMakie
using CairoMakie
using LinearAlgebra: normalize
using StatsBase: StatsBase, Histogram, fit, AbstractWeights, mean, Weights, median, mean_and_std, quantile
using JSON
using BSON
using DataFrames
using CategoricalArrays: categorical, levelcode, levels
using OnlineStats: OrderedDict
using GLM
using Format

GLMakie.activate!()

const GIT_ROOT = strip(read(`git rev-parse --show-toplevel`, String))

TOKENIZERS = OrderedDict(
        "smirk" => "smirk",
        "character" => "Character",
        "smirk-gpe-50k-nmb-ss" => "smirk-gpe",
        "smirk-gpe-50k-mb-ss" => "smirk-gpe, merge-brackets",
        "seyonec/ChemBERTa-zinc-base-v1" => "ChemBERTa v1",
        "ChangwenXu98/TransPolymer" => "TransPolymer",
        "ibm/MoLFormer-XL-both-10pct-oov" => "MoLFormer",
        "HUBioDataLab/SELFormer" => "SELFormer",
        "devalab/molgpt-moses" => "MolGPT-MOSES",
        "devalab/molgpt-guacamol" => "MolGPT-GuacaMol",
        "rxn4chemistry/rxn_yields" => "Yield-BERT",
        "rxn4chemistry/rxnfp" => "RXNFP",
        "MolecularAI/Chemformer" => "Chemformer",
        "MolecularAI/Chemformer-downstream" => "Chemformer, Downstream",
        "SmilesPE/SPE_ChEMBL" => "SmilesPE",
        "sagawa/ReactionT5-product-prediction" => "ReactionT5, Products",
        "sagawa/ReactionT5-yield-prediction" => "ReactionT5, Yield",
        "meta-llama/Meta-Llama-3.1-8B" => "Llama 3.1",
        "meta-llama/Meta-Llama-3-8B" => "Llama 3",
        "Xenova/gpt-4o" => "GPT-4o",
        "google/gemma-7b" => "Gemma",
        "google/gemma-2-2b" => "Gemma 2",
)

const CLASS_MARKER = Dict(
    :atomic => :x,
    :nlp => :diamond,
    :nlp_based => :cross,
    :ours => :star5,
)

function theme()
    Theme(
        rowgap = 5,
        colgap = 5,
        fonts = (;
            :regular => TokenizerStats.findfont("Helvetica", "Regular"),
            :bold => TokenizerStats.findfont("Helvetica", "Bold"),
        ),
        fontsize = 8,
        size = (246, 152),
        figure_padding = (1, 15, 2, 2),
        CairoMakie = (;
            pt_per_unit=1,
            px_per_unit=600/72
        ),
        GLMakie = (;
            px_per_unit=300/72,
            scalefactor=4,
        ),
        palette=(;
            color=cgrad(:seaborn_muted, 10),
            linestyle = [:solid, :dot, :dashdot],
        ),
        Lines = (;
            cycle = Cycle([:color, :linestyle], covary = true),
        ),
        Axis = (;
            titlesize = 6,
            titlegap = 1,
            spinewidth = 0.5,
            yticksize = 3,
            ytickwidth = 0.5,
            yminortickwidth = 0.25,
            yminorticksize = 2,
            xtickwidth = 0.5,
            xticksize = 3,
            xminortickwidth = 0.25,
            xminorticksize = 2,
            xgridwidth = 0.5,
            ygridwidth = 0.5,
            xminorgridwidth = 0.25,
            yminorgridwidth = 0.25,
        ),
        Legend = (;
            titlegap = 0,
            patchsize = (6, 6),
            rowgap = 0,
            colgap = 8,
            groupgap = 4,
            padding = 2,
            framewidth = 0.5,
            margin = (2, 2, 2, 2),
            tellheight = false,
            tellwidth = false,
        ),
        Colorbar = (;
            size = 8,
            spinewidth = 0.5,
            tickwidth = 0.5,
            ticksize = 2,
            labelpadding = 0,
            ticklabelpad = 0,
        ),
        Scatter = (;
            markersize = 8,
        )
    )
end


""" Save duplicate figures for publication and web """
function savefig(name::String, f::Figure)
    save(joinpath(@__DIR__, "fig", name * ".pdf"), f; pt_per_unit=1)
    save(joinpath(@__DIR__, "fig", name * ".png"), f; px_per_unit=600/72)
    return nothing
end

function all_figures()
    mkpath(joinpath(@__DIR__, "fig"))
    CairoMakie.activate!()

    with_theme(theme()) do
        # Token Usage
        savefig("token_usage", figure_token_usage())
        savefig("fertility", figure_fertility())
        savefig("oov_rate", figure_oov_rate())
        savefig("jaccard", figure_jaccard())

        # N-Gram Analysis
        savefig("ngram_fits", figure_ngram_fits())
        savefig("ngram_unk_log_odds", figure_ngram_info_loss())
        savefig("kl_v_info_loss", figure_kl_v_info_loss())
        savefig("info_loss_ref_tokenzier", figure_info_loss_ref_tokenizer())
        savefig("ngrma_unk_log_odds_cobalt", figure_ngram_info_loss(;
            smi="[Cl-][Co+2@OH1]([Cl-])([NH3])([NH3])([NH3])[NH3]",
            token_colors=("[NH3]" => :magenta, "[Co+2@OH1]" => :turquoise, "[Cl-]" => :orange)
        ))

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
    x = map(Base.Fix1(parse, Int)∘string, collect(keys(counts)))
    w = Int.(collect(values(counts))) |> StatsBase.FrequencyWeights
    nbins = hist_nbins(:sturges, x, w)
    h = fit(Histogram, x, w; nbins, kwargs...)
    h = normalize(h)
    centers = StatsBase.midpoints(first(h.edges))
    return centers, h.weights
end

function collect_results(glob=r"stats-.*\.json")
    tokenizers = Dict{String, String}()
    for result in find(joinpath(@__DIR__, "stats"), glob)
        name = joinpath(splitpath(relpath(result, @__DIR__))[2:end-1])
        tokenizers[name] = result
    end
    return tokenizers
end

function figure_token_usage()
    f = Figure(;
        size=72 .* (4.5, 3),
        figure_padding=(1, 1, 1, 5),
    )
    l = 1e-5
    ax = Axis(f[1,1];
        limits=((1, 2500), (0, 25)),
        xlabel="Token Rank",
        ylabel="Information Content [nats]",
        xscale=log10,
        xminorticksvisible=true,
        xminorticks=IntervalsBetween(5),
        xminorgridvisible=true,
    )

    results = collect_results(r"realspace_v4_dev2?\.bson")
    plt_tokenizers = [
        "smirk",
        "smirk-gpe-50k-nmb-ss",
        "character",
        "MolecularAI/Chemformer",
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
    rows = []
    for tok_name in plt_tokenizers
        if tok_name in keys(results)
            stats = BSON.load(results[tok_name])
            usage = stats[:train][:ngrams][1]
            usage = Dict{Int, Int}(only(k) => v for (k, v) in pairs(usage))
            vocab_size = stats[:tokenizer][:vocab_size]
            c_token = TokenizerStats.collate_token_usage(usage, vocab_size; smoothing=0)
            p_token = c_token ./ sum(c_token)

            sort!(p_token, rev=true)
            efficiency = sum(p -> -p * log(p)/log(vocab_size), filter(>(0), p_token))
            H = sum(p -> p > 0 ? -p * log(p) : 0, p_token; init=0.0)
            plt_name = TOKENIZERS[tok_name]
            push!(rows, (; tok_name, plt_name, usage, H, efficiency, vocab_size))
        end
    end
    df =DataFrame(rows)
    sort!(df, [:H, :efficiency, :vocab_size]; rev=true)
    display(select(df, Not(:usage)))

    linestyles = [:dot, :dashdot, :solid]
    for (idx, row) in enumerate(eachrow(df))
        tokenusage!(ax, row.usage, row.vocab_size;
            smoothing=0,
            label=row.plt_name * format(" ({:.2f}, {:.1%})", row.H, row.efficiency),
            linestyle=linestyles[idx % length(linestyles) + 1],
            linewidth=0.75,
        )
    end


    Legend(f[1,2], ax; tellheight=true, tellwidth=true, patchsize = (10, 6), valign=:top)
    colgap!(f.layout, 2)
    resize_to_layout!(f)
    return f
end


function figure_vocab_entropy(results::Dict)
    f = Figure(size=(600, 300))
    ax = Axis(f[1,1];
        ylabel="Entropy [bits]",
        limits=(nothing, (-2.5, 25)),
        xticklabelrotation = 0.4,
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
    violin!(ax, categories, values; weights = weights, show_median=true)
    ax.xticks[] = (1:length(names), names)

    return f
end

function figure_fertility()
    @info "Plotting Cross Entropy Loss vs. Fertility"
    f = Figure(size=72 .* (3.42, 2.75))
    ax = Axis(f[1,1];
        limits=((0, 75), nothing),
        ylabel="Cross Entropy Loss",
        xlabel="Tokenizer Fertility",
    )
    usage_stats = TokenizerStats.usage_stats()
    loss_stats = TokenizerStats.model_loss_stats()
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

    # Select the best n-gram model for each tokenizer
    val_loss = subset(loss_stats,
        :split => ByRow(==(:val)),
        :dataset => ByRow(==("realspace_v4_dev")),
        :training_dataset => ByRow(==("realspace_v4_dev")),
        :tokenizer => ByRow(x -> x in plt_tokenizers),
    )
    best_models = combine(groupby(val_loss, :tokenizer)) do gdf
        sort!(gdf, :avg_model_loss; rev=false)
        return gdf[1, :]
    end

    fertility = subset(usage_stats,
        :split => ByRow(==(:train)),
        :dataset => ByRow(==("realspace_v4_dev")),
    )
    select!(fertility, [:tokenizer, :avg_fertility, :std_fertility])

    df = leftjoin(best_models, fertility, on=[:tokenizer])
    df.class = classify_tokenizer.(df.tokenizer)
    sort!(df, [:class, :avg_model_loss])
    dropmissing!(df)
    display(df)

    # Regression modle
    ols = lm(@formula(avg_model_loss ~ avg_fertility + 1), df)
    display(ols)

    # Plot results
    ablines!(ax, coef(ols)[1], coef(ols)[2]; label="Linear Least Squares Fit", color=:red, linewidth=0.5)
    for row in eachrow(df)
        name = row.tokenizer
        label = haskey(TOKENIZERS, name) ? TOKENIZERS[name] : name
        marker = CLASS_MARKER[classify_tokenizer(name)]
        scatter!(ax, row.avg_fertility, row.avg_model_loss; label, marker)
    end

    Legend(f[1,2], ax; tellheight=true, tellwidth=true, valign=:top)
    # axislegend(ax, position=:lb, fontsize=8)
    resize_to_layout!(f)

    return f
end

collate_atomic_oov(key::String, results::Dict) = results[key]["oov"] / results[key]["nobs"]
function collate_atomic_oov(key::Regex, results::Dict)
    nobs = 0
    oov = 0
    for (k, v) in results
        if !isnothing(match(key, k))
            nobs += v["nobs"]
            oov += v["oov"]
        end
    end
    return oov / nobs
end



function figure_oov_rate(filename=joinpath(@__DIR__, "stats-atomic.json"))
    datasets = OrderedDict(
        "Elements" => "elements",
        "Bonds" => "bonds",
        "Isotopes" => "isotopes",
        "Carbon Rings" => "rings",
        "Ions" => "charged_elements",
        "Chirality" => "chiral_elements",
        "Charged, Chiral Isotopes" => "charged_chiral_isotopes",
        "MoleculeNet" => r"^MoleculeNet/",
    )

    plt_tokenizers = [
        "smirk",
        "smirk-gpe-50k-nmb-ss",
        # "character",
        "seyonec/ChemBERTa-zinc-base-v1",
        "ChangwenXu98/TransPolymer",
        "ibm/MoLFormer-XL-both-10pct-oov",
        "HUBioDataLab/SELFormer",
        "devalab/molgpt-moses",
        "devalab/molgpt-guacamol",
        # "rxn4chemistry/rxn_yields"
        "rxn4chemistry/rxnfp",
        "MolecularAI/Chemformer",
        "SmilesPE/SPE_ChEMBL",
        "sagawa/ReactionT5-product-prediction",
        "sagawa/ReactionT5-yield-prediction",
    ]
    f = Figure(;
        size=72 .* (7, 2),
        figure_padding=(1, 1, 1, 5),
    )
    ax = Axis(f[1,1];
        ylabel="Out of Vocab Rate",
        limits=(nothing, (0, 100)),
        ytickformat="{:.0f}%",
        xticklabelrotation = 0.4,
        xticks=(1:length(plt_tokenizers),  map(n -> TOKENIZERS[n], plt_tokenizers)),
        xticklabelsize=10,
        yminorticks=IntervalsBetween(5),
        yminorticksvisible=true,
        yminorgridvisible = true,
        xgridvisible = false,
    )

    data = JSON.parsefile(filename)
    tok_pos = Int[]
    ds_group_pos = Int[]
    oov_rate = Float64[]
    for (idx, tok_name) in enumerate(plt_tokenizers)
        tok_name = startswith(tok_name, "smirk-gpe") ? "./" * tok_name : tok_name
        tok_results = data[tok_name]
        for (gdx, group_key) in enumerate(collect(values(datasets)))
            group_oov_rate = collate_atomic_oov(group_key, tok_results) * 100
            push!(tok_pos, idx)
            push!(ds_group_pos, gdx)
            push!(oov_rate, group_oov_rate)
        end
    end

    h = barplot!(ax, tok_pos, oov_rate;
                 dodge=ds_group_pos,
                 color=ds_group_pos,
                 gap=0.1,
                 colorrange=(1, length(datasets)),
                 colormap=:Set1_8,
                 strokewidth=0.25,
                 strokecolor=:black,
    )
    ds_elements = map(1:length(datasets)) do gdx
        PolyElement(polycolor=gdx, colormap=h.colormap, colorrange=h.colorrange)
    end
    Legend(f[1,2], ds_elements, collect(keys(datasets));
        tellheight=true, tellwidth=true, orientation=:vertical,
        patchstrokecolor=:black,
        patchstrokewidth=1,
        framevisible=false,
        padding=0,
    )

    resize_to_layout!(f)
    return f
end

function figure_ngram_fits(; colormap=:Set2_5)
    df = TokenizerStats.model_loss_stats()
    plt_toks = [
        "smirk",
        "smirk-gpe-50k-nmb-ss",
        "character",
        "ibm/MoLFormer-XL-both-10pct-oov",
        "devalab/molgpt-moses",
        # "rxn4chemistry/rxn_yields",
        "rxn4chemistry/rxnfp",
        "SmilesPE/SPE_ChEMBL",
        "MolecularAI/Chemformer",
        "seyonec/ChemBERTa-zinc-base-v1",
        "ChangwenXu98/TransPolymer",
        "meta-llama/Meta-Llama-3.1-8B",
        "Xenova/gpt-4o",
        "google/gemma-7b",
    ]
    subset!(df,
        :tokenizer => ByRow(x -> x in plt_toks),
        :split => ByRow(==(:val)),
    )
    df.tokenizer = categorical(df.tokenizer; levels=plt_toks)

    # Set up figure
    f = Figure(; size=72 .* (7, 3))
    tokenizer = levels(df.tokenizer)
    ax_kwargs = (;
        limits=(nothing, (0, nothing)),
        ylabelsize=7,
        xticks = (1:length(tokenizer), map(n -> TOKENIZERS[n], plt_toks)),
        xticklabelrotation = 0.4,
        xticksvisible=false,
        xgridvisible=false,
    )
    ax_pretrain = Axis(f[1,1];
        ylabel="Enimine REAL Space\nCross Entropy Loss [nats]",
        ax_kwargs...
    )
    hidexdecorations!(ax_pretrain)
    ax_molnet = Axis(f[2,1];
        ylabel="MoleculeNet\nCross Entropy Loss [nats]",
        ax_kwargs...
    )

    # N-Gram Legend
    ds_elements = map(1:5) do gdx
        PolyElement(polycolor=gdx;
            colormap,
            colorrange=(1, 5),
        )
    end
    Legend(f[0,1], ds_elements, ["unigram", "bigram", "trigram", "4-gram", "5-gram"];
        tellheight=true, tellwidth=false, orientation=:horizontal,
        framevisible=false,
        patchstrokecolor=:black,
        patchstrokewidth=1,
        margin=(0, 0, 0, 0),
        padding=1,
    )

    # Pretraining
    df_pretrain = subset(df,
        :training_dataset => ByRow(==("realspace_v4_dev")),
        :dataset => ByRow(==("realspace_v4_dev")),
    )
    barplot!(ax_pretrain, levelcode.(df_pretrain.tokenizer), df_pretrain.avg_model_loss;
        dodge=df_pretrain.ngram,
        color=df_pretrain.ngram,
        colormap,
        fillto=1,
    )

    # Finetune
    df_finetune = subset(df,
        :training_dataset => ByRow(==("realspace_v4_dev")),
        :dataset => ByRow(!=("realspace_v4_dev")),
    )
    df_finetune = combine(groupby(df_finetune, [:tokenizer, :ngram])) do gdf
        return (;
            avg_model_loss = mean(gdf.avg_model_loss, Weights(gdf.samples)),
        )
    end
    barplot!(ax_molnet, levelcode.(df_finetune.tokenizer), df_finetune.avg_model_loss;
        dodge=df_finetune.ngram,
        color=df_finetune.ngram,
        colormap,
        fillto=1,
    )
    rowgap!(f.layout, 2)
    resize_to_layout!(f)
    return f
end

function figure_ngram_prediction(smi; direction=:forward)
    f = Figure(size=72 .* (3.42, 3))
    cb = Colorbar(f[1:3, 4];
        label="Log Probability",
        colormap=:lipari,
        colorrange=(-10, 0),
        tickformat="{:2d}",
    )

    path(name) = (joinpath(@__DIR__, "stats", name, "realspace_v4_dev.bson"), TOKENIZERS[name])
    tok_log_prob!(f[1,1], cb, path("smirk")..., smi; direction)
    tok_log_prob!(f[1,2], cb, path("ibm/MoLFormer-XL-both-10pct-oov")..., smi; direction)
    tok_log_prob!(f[1,3], cb, path("seyonec/ChemBERTa-zinc-base-v1")..., smi; direction)
    tok_log_prob!(f[2,1], cb, path("devalab/molgpt-moses")..., smi; direction)
    tok_log_prob!(f[2,2], cb, path("rxn4chemistry/rxnfp")..., smi; direction)
    tok_log_prob!(f[2,3], cb, path("MolecularAI/Chemformer")..., smi; direction)
    tok_log_prob!(f[3,1], cb, path("meta-llama/Meta-Llama-3.1-8B")..., smi; direction)
    tok_log_prob!(f[3,2], cb, path("Xenova/gpt-4o")..., smi; direction)
    tok_log_prob!(f[3,3], cb, path("google/gemma-7b")..., smi; direction)

    # Format plot
    Label(f[:, 0], smi, rotation=pi/2, fontsize=length(smi) > 40 ? 6 : 8, padding=(0, 2, 0, 0))
    Label(f[end+1, :], "Predicted Tokens", fontsize=8)
    resize_to_layout!(f)
    rowgap!(f.layout, 1)
    colgap!(f.layout, 1)

    return f
end

function tok_log_prob!(f, cb, file, name, smi, max_vocab=200; direction=:forward)
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
    P = P[vocab .+ 1, :]

    # Truncate Vocab
    max_vocab = min(max_vocab, size(P, 1))
    name = size(P, 1) > max_vocab ? "*" * name : name
    sdx = sortperm(vec(sum(P; dims=2)); rev=true)
    P = P[sdx[1:max_vocab], :]

    ax = Axis(f;
        title = "$(name): $(round(l; sigdigits=2))",
        limits=((1, size(P, 1)), (1, size(P, 2))),
        xticksvisible = false,
        xticklabelsvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        spinewidth=0.5,
        aspect=1,
    )
    h = heatmap!(ax, P; colormap=cb.colormap, colorrange=cb.colorrange)

    # # Highlight the correct token
    # for (code_pos, token_id) in enumerate(code)
    #     box_token!(ax, token_id, code_pos; linewidth=1.0, color=:green)
    # end

    return nothing
end

function figure_ngram_info_loss(;
    smi = "C(=Cc1ccccc1)C1=[O+][Cu-3]2([O+]=C(C=Cc3ccccc3)CC(c3ccccc3)=[O+]2)[O+]=C(c2ccccc2)C1",
    token_colors = ("[O+]" => :turquoise, "[Cu-3]" => :magenta)
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
    ref_file = joinpath(@__DIR__, "stats", "character", "realspace_v4_dev.bson")
    ngram, ref_tok, ref_info = TokenizerStats.load_ngram_model(ref_file)
    ref_code = pyconvert(Vector{Int}, ref_tok(smi)["input_ids"])
    kwargs = (; ngram, ref_tok, ref_code, token_color)

    tok_info_loss!(f[1,1], cb, "smirk", smi; kwargs...)
    # tok_info_loss!(f[1,2], cb, "ibm/MoLFormer-XL-both-10pct-oov", smi; kwargs...)
    tok_info_loss!(f[1,2], cb, "SmilesPE/SPE_ChEMBL", smi; kwargs...)
    # tok_info_loss!(f[2,1], cb, "MolecularAI/Chemformer", smi; kwargs...)
    tok_info_loss!(f[1,3], cb, "devalab/molgpt-moses", smi; kwargs...)
    # tok_info_loss!(f[2,3], cb, "rxn4chemistry/rxn_yields", smi; kwargs...)

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
        segment = get(colors, smi[m], cmap[length(colors)+1] )
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
    name = Dict(TOKENIZERS)[tok]
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
    i, P, Q = TokenizerStats.information_loss(ngram, ref_code, masked; N=2)

    # Remove special tokens
    vocab = TokenizerStats.nonspecial_vocab(ngram)
    unk = pyconvert(Int, ref_tok.unk_token_id)
    filter!(!=(unk), vocab)
    P = P[vocab .+ 1, :]
    Q = Q[vocab .+ 1, :]

    odds_ratio = @. (Q - log(1 - exp(Q))) - (P - log(1 - exp(P)))
    @info "masked for $name" masked_tokens=join(ref_tokens[masked], " ") extrema(odds_ratio)

    ax = Axis(f;
        title = "$name: $(round(i; sigdigits=3))",
        xticks = collect(5:5:length(vocab)),
        xticksvisible = false,
        xticklabelsvisible = false,
        yticksvisible = false,
        yticklabelsvisible = false,
        aspect=1,
        spinewidth=0.5,

    )
    heatmap!(ax, odds_ratio;
        colormap=cb.colormap,
        colorrange=cb.colorrange,
        highclip=cb.highclip,
        lowclip=cb.lowclip,
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
(m::Asinh)(x::Real) = m.a *asinh(x / m.a)

Makie.inverse_transform(m::Asinh) = x -> m.a * sinh(x / m.a)
Makie.defined_interval(::Asinh) = Makie.defined_interval(identity)
Makie.defaultlimits(m::Asinh) = (0.0, 10 * m.a)

Makie.inverse_transform(::typeof(asinh)) = sinh
Makie.defined_interval(::typeof(asinh)) = Makie.defined_interval(identity)
Makie.defaultlimits(::typeof(asinh)) = (0.0, 10.0)

function figure_kl_v_info_loss(; reference="character")
    model_loss = TokenizerStats.model_loss_stats()
    info_loss = TokenizerStats.avg_molnet_info_loss()

    subset!(model_loss,
        :dataset => ByRow(!=("realspace_v4_dev")),
        :training_dataset => ByRow(==("realspace_v4_dev")),
        :split => ByRow(==(:val)),
    )
    subset!(info_loss,
        :ref_tokenizer => ByRow(==(reference)),
    )
    model_loss = combine(groupby(model_loss, [:tokenizer, :split, :ngram])) do gdf
        return (;
            avg_model_loss = mean(gdf.avg_model_loss, Weights(gdf.samples)),
            vocab_size = first(gdf.vocab_size),
            samples = sum(gdf.samples),
        )
    end
    select!(model_loss, Not([:samples]))
    select!(info_loss, Not([:samples, :vocab_size]))
    df = leftjoin(model_loss, info_loss, on=[:tokenizer, :split, :ngram])
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
        :split => ByRow(==(:val)),
    )
    @info "Tokens with no info_loss" unique(df.tokenizer)
    subset!(df, :tokenizer => ByRow(x -> x in first.(plt_tokenizers)))
    df.tokenizer = categorical(df.tokenizer)
    display(df[!, [:tokenizer, :ngram, :avg_model_loss, :avg_info_loss]])

    f = Figure(; size=72 .* (4.5, 3), figure_padding=(1, 1, 1, 4))
    ax = Axis(f[1,1];
        limits=((50, 230), (-0.01, 64)),
        xlabel="Cross Entropy Loss [nats]",
        ylabel="Information Loss [nats]",
        yticks=[0, 0.05, 0.25, 1, 4, 16, 64],
        yscale=Asinh(0.05),
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
    ntokenizers = length(levels(df.tokenizer))
    tokenizer_elements = map(levels(df.tokenizer)) do name
        MarkerElement(; marker=Dict(plt_tokenizers)[name], color=:black)
    end
    tokenizer_labels = map(levels(df.tokenizer)) do name
        return TOKENIZERS[name]
    end
    Legend(f[1,1],
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

function figure_info_loss_ref_tokenizer()
    df = TokenizerStats.avg_molnet_info_loss()
    tokenizers = ("character", "meta-llama/Meta-Llama-3.1-8B")
    subset!(df,
        :split => ByRow(==(:val)),
        :ngram => ByRow(>(1)),
        :ref_tokenizer => ByRow(in(tokenizers)),
    )
    select!(df, [:tokenizer, :ref_tokenizer, :ngram, :avg_info_loss])
    df = unstack(df, [:tokenizer, :ngram], :ref_tokenizer, :avg_info_loss)
    dropmissing!(df)
    @info "Info Loss" characters=extrema(df.character) llama=extrema(df[!, "meta-llama/Meta-Llama-3.1-8B"])
    @info "Tokenizers" unique(df.tokenizer)
    display(subset(df, :ngram => ByRow(==(5))))

    f = Figure(; size=72 .* (6.5, 3))
    ax = Axis(f[1,1];
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
        marker = :x,
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
            avg_model_loss = mean(gdf.avg_model_loss, Weights(gdf.samples)),
        )
    end
    model_loss = unstack(model_loss, :ngram, :tokenizer, :avg_model_loss)
    dropmissing!(model_loss)

    ax = Axis(f[1,2];
        title = "Cross-Entropy Loss [nats]",
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
    Legend(f[1,2], [hab, ngram_elements...], [hab.label, ngram_labels...];
        tellheight=false,
        tellwidth=false,
        halign=:right,
        valign=:bottom,
        margin = (10, 10, 5, 10),
    )
    Label(f[1, 1, TopLeft()], "a)";
        font = :bold,
        halign = :right,
        padding = (0, 15, 5 , 0),
    )
    Label(f[1, 2, TopLeft()], "b)";
        font = :bold,
        halign = :right,
        padding = (0, 15, 5 , 0),
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
    ax = Axis(f[1,1];
        aspect=1,
        xticks=(1:length(k), k),
        yticks=(1:length(k), k),
        xticklabelrotation=pi/4,
        xticklabelsize=6,
        yticklabelsize=6,
    )
    h = heatmap!(ax, J; colormap=:lajolla, colorrange=(0, 1))
    Colorbar(f[1,2];
        tickformat="{:.0%}",
        label="Jaccard Index",
        colormap=h.colormap,
        colorrange=h.colorrange,
    )
    resize_to_layout!(f)
    return f
end
