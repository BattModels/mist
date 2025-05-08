module SmirkPaperPlots

using Makie
using OnlineStats
using StatsBase
using OnlineStats: Moments
using GLM
using DataFrames
using CairoMakie: CairoMakie
using Colors: distinguishable_colors, weighted_color_mean, RGBA
using CategoricalArrays: categorical
using Format: format
using FreeTypeAbstraction: FreeTypeAbstraction, newface, FTFont
using PythonCall: Py, pyconvert
using JLD2: jldopen
using JSON: JSON
using CategoricalArrays: categorical, levelcode
using HypothesisTests: HypothesisTests, HypothesisTest, VarianceEqualityTest, pvalue
using Distributions: Chisq, FDist, Normal
using OrderedCollections: OrderedDict
using CSV: CSV
using Clustering: hclust
using RegressionTables: regtable, LatexTable

using TokenizerStats
using TokenizerStats: load_tokenizer, find

# Tabulate Results for plotting / analysis
include("tabulate_results.jl")
include("tabulate_tokenizer.jl")
include("tokenizer_summary.jl")
include("prognostics.jl")

# Helper for plotting
include("plot_utils.jl")
include("coeff_plot.jl")

# Figures
include("figures/info_loss.jl")
include("figures/ngram.jl")
include("figures/transfromer.jl")
include("figures/jaccard.jl")
include("figures/fe_models.jl")
include("figures/intrinsic.jl")


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
    data = JSON.parsefile(abspath(joinpath(stats_dir, "..", "tokenizers.json")))
    return Dict(tok["name_or_path"] => tok for tok in data)
end


""" Save duplicate figures for publication and web """
function savefig(name::String, f::Figure; dpi=300)
    fig_dir = joinpath(pkgdir(TokenizerStats), "fig")
    mkpath(fig_dir)
    save(joinpath(fig_dir, name * ".pdf"), f; pt_per_unit=1, backend=CairoMakie)
    save(joinpath(fig_dir, name * ".png"), f; px_per_unit=dpi / inch, backend=CairoMakie)
    return nothing
end

end
