using TokenizerStats: TokenizerStats, tokenizer_label, moments!, hist_nbins, find, tokenusage!
using GLMakie
using LinearAlgebra: normalize
using StatsBase: StatsBase, Histogram, fit, AbstractWeights
using JSON
using DataFrames

const GIT_ROOT = strip(read(`git rev-parse --show-toplevel`, String))

TOKENIZERS = [
        "smirk" => "smirk",
        "smirk-gpe-50k-mb-ss" => "smirk-gpe-50k-mb-ss",
        "seyonec/ChemBERTa-zinc-base-v1" => "ChemBERTa v1",
        # "ChangwenXu98/TransPolymer" => "TransPolymer",
        "ibm/MoLFormer-XL-both-10pct-oov" => "MoLFormer",
        "HUBioDataLab/SELFormer" => "SELFormer",
        # "devalab/molgpt-moses" => "MolGPT, moses",
        "devalab/molgpt-guacamol" => "MolGPT, guacamol",
        # "rxn4chemistry/rxn_yields" => "Yield-BERT",
        "rxn4chemistry/rxnfp" => "RXNFP",
        "MolecularAI/Chemformer" => "Chemformer",
        "SmilesPE/SPE_ChEMBL" => "SmilesPE",
        # "sagawa/ReactionT5-product-prediction" => "ReactionT5, Products",
        # "sagawa/ReactionT5-yield-prediction" => "ReactionT5, Yield",
]


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

function figure_token_usage(results::Dict)
    f = Figure(; size=(500, 300))
    l = 1e-5
    ax = Axis(f[1,1];
        limits=((0, 1), (0, nothing)),
        xlabel="Token Usage Rank [%]",
        ylabel="Information Content [nats]",
        xtickformat="{:.0%}",
    )

    for (tok_name, plt_name) in TOKENIZERS
        if tok_name in keys(results)
            stats = JSON.parsefile(results[tok_name])
            tokenusage!(ax, stats; label=plt_name)
        end
    end

    Legend(f[1,2], ax; tellwidth=true, tellheight=false, nbanks=1,
           orientation=:vertical, valign=:top, framevisible=false,
           rowgap=1, labelsize=10, patchsize=(10, 10),
    )
    colgap!(f.layout, 1)
    resize_to_layout!(f)
    return f
end


function figure_ignorance(results::Dict)
    f = Figure()
    ax = Axis(f[1,1])
    tok_names = String[]
    tok_entropy = Float64[]
    tok_unk_info = Float64[]

    for (tok_name, plt_name) in TOKENIZERS
        tok_name in keys(results) || continue
        tok_stats = JSON.parsefile(results[tok_name])

        # Compute Corpus Entropy
        vocab_size = Int(tok_stats["tokenizer"]["vocab_size"])

        # Check max token id in usage, add 1 to account for zero indexing
        vocab_size = max(vocab_size, 1+maximum(parse.(Int, keys(tok_stats["token_usage"]))))
        usage = TokenizerStats.collate_token_usage(0:vocab_size-1, tok_stats["token_usage"]; smoothing=1)
        token_prob = usage ./ sum(usage)
        token_entropy = @. -token_prob * log2(token_prob)
        entropy = sum(token_entropy; init=0.0)

        unk_token_id = Int(tok_stats["tokenizer"]["unk_token_id"])
        push!(tok_names, plt_name)
        push!(tok_entropy, entropy)
        push!(tok_unk_info, token_entropy[unk_token_id+1])
    end
    scatter!(ax, tok_entropy, tok_unk_info)
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

function figure_fertility(results::Dict)
    f = Figure(size=(900, 500))
    ax = Axis(f[1,1];
        ylabel="Fertility",
        limits=(nothing, (0, nothing)),
    )
    ax_unique = Axis(f[1,2];
        ylabel="Number of Unique Tokens",
        limits=(nothing, (0, nothing)),
    )
    hidexdecorations!(ax)
    hidexdecorations!(ax_unique)
    kwargs = (; show_notch=true, show_outliers=false)
    for (idx, (name, file)) in enumerate(pairs(results))
        stats = JSON.parsefile(file)
        values, weights = fit_hist(stats["fertility"])
        boxplot!(ax, repeat([idx], length(values)), values; weights, label=name, kwargs...)
        values, weights = fit_hist(stats["nunique"])
        boxplot!(ax_unique, repeat([idx], length(values)), values; weights, label=name, kwargs...)
    end
    Legend(f[2,:], ax; tellwidth=true, tellheight=true, nbanks=3)


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



function figure_oov_rate(filename)
    datasets = [
        "Elements" => "elements",
        "Bonds" => "bonds",
        "Isotopes" => "isotopes",
        "Carbon Rings" => "rings",
        "Ions" => "charged_elements",
        "Chirality" => "chiral_elements",
        "Charged, Chiral Isotopes" => "charged_chiral_isotopes",
        "MoleculeNet" => r"^MoleculeNet/",
    ]

    tokenizers = [
        "smirk" => "smirk",
        "seyonec/ChemBERTa-zinc-base-v1" => "ChemBERTa v1",
        "ChangwenXu98/TransPolymer" => "TransPolymer",
        "ibm/MoLFormer-XL-both-10pct-oov" => "MoLFormer",
        "HUBioDataLab/SELFormer" => "SELFormer",
        "devalab/molgpt-moses" => "MolGPT, moses",
        "devalab/molgpt-guacamol" => "MolGPT, guacamol",
        "rxn4chemistry/rxn_yields" => "Yield-BERT",
        "rxn4chemistry/rxnfp" => "RXNFP",
        "MolecularAI/Chemformer" => "Chemformer",
        "SmilesPE/SPE_ChEMBL" => "SmilesPE",
        "sagawa/ReactionT5-product-prediction" => "ReactionT5, Products",
        "sagawa/ReactionT5-yield-prediction" => "ReactionT5, Yield",
    ]
    f = Figure(size=(600, 300))
    ax = Axis(f[1,1];
        ylabel="Out of Vocab Rate [%]",
        limits=(nothing, (0, 100)),
        ytickformat="{:.0f}%",
        xticklabelrotation = 0.4,
        xticks=(1:length(tokenizers), last.(tokenizers)),
        xticklabelsize=10,
    )

    data = JSON.parsefile(filename)
    tok_pos = Int[]
    ds_group_pos = Int[]
    oov_rate = Float64[]
    for (idx, name) in enumerate(first.(tokenizers))
        tok_results = data[name]
        for (gdx, group_key) in enumerate(last.(datasets))
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
    Legend(f[1,2], ds_elements, collect(first.(datasets));
        tellheight=true, tellwidth=true, orientation=:vertical,
        nbanks=1, labelsize=10, patchsize=(10,10), rowgap=3, colgap=4,
        framevisible=false,
        patchstrokecolor=:black, patchstrokewidth=1,
    )
    rowgap!(f.layout, 5)
    colgap!(f.layout, 5)

    resize_to_layout!(f)
    return f
end

function figure_equal_odds()
    df_runs, df_metrics = TokenizerStats.summarize_finetuning(joinpath(".cache", "wandb-export", "finetuning"))
    subset!(df_runs,
        :tags => ByRow(tags -> "sweep-aug15-1" in tags),
        :state => ByRow(==("finished")),
    )
    df_metrics = innerjoin(df_metrics, df_runs[!, [:id]]; on=:id)

    # Compute Equal Odds
    xtab = TokenizerStats.widden_crosstab(df_metrics)
    eq_odd_diff = combine(groupby(xtab, [:split, :id])) do gdf
        n_oov = only(gdf[gdf.tok_group .== "oov", :nobs])
        n_non_oov = only(gdf[gdf.tok_group .== "non_oov", :nobs])

        # Prevalence
        n_obs = only(gdf[gdf.tok_group .== "all", :nobs])
        tp = only(gdf[gdf.tok_group .== "all", :tp])
        fn = only(gdf[gdf.tok_group .== "all", :fn])
        prevalence = (tp + fn) / n_obs

        # Misclassification Spread
        gdf = gdf[gdf.tok_group .!= "all", :]
        fpr_range = maximum(1 .- gdf.tpr) - minimum(1 .- gdf.tpr)
        fnr_range = maximum(1 .- gdf.tnr) - minimum(1 .- gdf.tnr)

        # If only one group is present, equal_odds_diff is undef
        if n_oov == 0 || n_non_oov == 0
            equal_odds_diff = missing
        else
            equal_odds_diff = max(fpr_range, fnr_range)
        end
        (; equal_odds_diff, prevalence)
    end

    # Add Dataset
    equal_odds_diff = leftjoin(eq_odd_diff, df_runs[!, [:id, :dataset, :tokenizer]]; on=:id)
    equal_odds_diff.tokenizer .= replace.(equal_odds_diff.tokenizer, r".*\.ckpt" => "smirk")
    sort!(equal_odds_diff, [:dataset, :tokenizer, :split])
    equal_odds_diff

end
