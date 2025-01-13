module SmirkPaperPlots

using Makie
using OnlineStats
using OnlineStats: Moments
using GLM
using DataFrames
using CairoMakie: CairoMakie
using Colors: distinguishable_colors, weighted_color_mean, RGBA
using Format: format
using PythonCall: Py, pyconvert
using StatsBase: StatsBase, mean, stderr, mean_and_std, AbstractWeights, Weights
using FreeTypeAbstraction: FreeTypeAbstraction, newface, FTFont
using JLD2: jldopen
using JSON: JSON
using CategoricalArrays: categorical, levelcode
using HypothesisTests: HypothesisTests, HypothesisTest, VarianceEqualityTest, pvalue
using Distributions: Chisq, FDist, Normal
using OrderedCollections: OrderedDict
using CSV: CSV

using TokenizerStats
using TokenizerStats: load_tokenizer, find

# Tabulate Results for plotting / analysis
include("tabulate_results.jl")
include("tabulate_tokenizer.jl")

# Helper for plotting
include("plot_utils.jl")

# Figures
include("figures/info_loss.jl")
include("figures/ngram.jl")
include("figures/transfromer.jl")
include("figures/jaccard.jl")


const CLASS_MARKER = Dict(
    :atomic => :x,
    :nlp => :diamond,
    :nlp_based => :cross,
    :ours => :star5,
)

CLASS_PLT_LABEL = Dict(
    "smirk" => "Smirk",
    "smirk-gpe" => "Smirk-GPE",
    "spe" => "SPE",
    "bpe" => "BPE",
    "unigram" => "Unigram",
    "atomwise" => "Atom-wise",
)

# Figure Units
const pt = 3 / 4
const inch = 96

function tokenizers_info(stats_dir)
    data = JSON.parsefile(joinpath(stats_dir, "tokenizers.json"))
    return Dict(tok["name_or_path"] => tok for tok in data)
end


""" Save duplicate figures for publication and web """
function savefig(name::String, f::Figure; dpi=300)
    fig_dir = joinpath(pkgdir(TokenizerStats), "fig")
    mkpath(fig_dir)
    save(joinpath(fig_dir, name * ".pdf"), f; pt_per_unit=1)
    save(joinpath(fig_dir, name * ".png"), f; px_per_unit=dpi / inch)
    return nothing
end

function (@main)(stats_dir=joinpath(pkgdir(TokenizerStats), "stats"))
    CairoMakie.activate!()

    # Tokenizer Statistics
    loss_stats = model_loss_stats(stats_dir)
    info_loss = info_loss_stats(stats_dir)
    token_usage = usage_stats(stats_dir)

    # Transformer Models
    dfp, dff, dft = transformer_models(stats_dir;
        sweep_file=joinpath(pkgdir(TokenizerStats), "pipeline_unfrozen.json")
    )

    with_theme(theme()) do
        # Token Usage
        savefig("token_usage", figure_token_usage(stats_dir))
        savefig("oov_rate", figure_oov_rate(stats_dir))
        # savefig("jaccard", figure_jaccard())

        # Transformers vs. N-Grams
        df = ngram_vs_transformer_fits(stats_dir, loss_stats, dfp)
        savefig("ngram_vs_transformer", figure_ngram_vs_transformer(stats_dir, df))

        # Transformer Model Summary
        f, df = figure_tf_finetune(stats_dir, dff, dft)
        savefig("tf_finetune", f)
        CSV.write(joinpath("stats/tf_model_summary.csv"), df)

        # N-Gram Analysis
        savefig("ngram_fits", figure_ngram_fits(loss_stats, stats_dir))
        # savefig("ngram_unk_log_odds", figure_ngram_info_loss())
        savefig("kl_v_info_loss", figure_kl_v_info_loss(stats_dir, loss_stats, info_loss))
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
                savefig("log_prob_$(direction)_$name", figure_ngram_prediction(smi, stats_dir; direction))
            end
        end
    end
end

end
