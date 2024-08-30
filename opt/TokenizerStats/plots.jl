using TokenizerStats: TokenizerStats, tokenizer_label, moments!, hist_nbins, find, tokenusage!
using PythonCall
using GLMakie
using CairoMakie
using LinearAlgebra: normalize
using StatsBase: StatsBase, Histogram, fit, AbstractWeights, mean, Weights, median, mean_and_std
using JSON
using BSON
using DataFrames
using CategoricalArrays: categorical, levelcode, levels
using OnlineStats: OrderedDict

GLMakie.activate!()

const GIT_ROOT = strip(read(`git rev-parse --show-toplevel`, String))

TOKENIZERS = OrderedDict(
        "smirk" => "smirk (ours)",
        "character" => "Character",
        "smirk-gpe-50k-mb-ss" => "smirk-gpe-50k-mb-ss",
        "seyonec/ChemBERTa-zinc-base-v1" => "ChemBERTa v1",
        "ChangwenXu98/TransPolymer" => "TransPolymer",
        "ibm/MoLFormer-XL-both-10pct-oov" => "MoLFormer",
        "HUBioDataLab/SELFormer" => "SELFormer",
        "devalab/molgpt-moses" => "MolGPT, moses",
        "devalab/molgpt-guacamol" => "MolGPT, guacamol",
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
)

function theme()
    Theme(
        rowgap = 5,
        colgap = 5,
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
            linestyle = [:solid, :dash],
        ),
        Lines = (;
            cycle = [:color, :linestyle],
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

        # N-Gram Analysis
        savefig("ngram_fits", figure_ngram_fits())
        savefig("ngram_unk_log_odds", figure_ngram_info_loss())
        savefig("kl_v_info_loss", figure_kl_v_info_loss())

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
    f = Figure(; size=72 .* (4.5, 3))
    l = 1e-5
    ax = Axis(f[1,1];
        limits=((0, 1), (0, 37)),
        xlabel="Token Usage Rank [%]",
        ylabel="Information Content [nats]",
        xtickformat="{:.0%}",
        xscale=sqrt,
    )

    results = collect_results(r"realspace_v4_dev\.bson")
    for (tok_name, plt_name) in TOKENIZERS
        if tok_name in keys(results)
            stats = BSON.load(results[tok_name])
            usage = stats[:ngrams][1]
            usage = Dict{Int, Int}(only(k) => v for (k, v) in pairs(usage))
            vocab_size = stats[:tokenizer][:vocab_size]
            tokenusage!(ax, usage, vocab_size; label=plt_name, smoothing=0)
        end
    end

    axislegend(ax, position=:rt, nbanks=1)
    colgap!(f.layout, 1)
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
    f = Figure(size=72 .* (4, 3.25))
    ax = Axis(f[1,1];
        limits=((0, 75), nothing),
        ylabel="Unigram KL-Divergence",
        xlabel="Tokenizer Fertility",
    )
    usage_stats = collect_results(r"realspace_v4_dev\.bson")
    loss_stats = TokenizerStats.model_loss_stats()
    plt_tokenizers = filter(x -> x != "character", collect(keys(TOKENIZERS)))
    @info plt_tokenizers

    # Average Loss over MoleculeNet for unigrams on the validation set
    subset!(loss_stats,
        :ngram => ByRow(==(1)),
        :split => ByRow(x -> string(x) == "val"),
        :tokenizer => ByRow(x -> x in plt_tokenizers),
    )
    @assert nrow(loss_stats) > 1
    @warn "using median loss, replace with weighted-mean"
    loss_stats = combine(groupby(loss_stats, :tokenizer)) do gdf
        # μ, σ = mean_and_std(gdf.avg_model_loss, Weights(gdf.samples))
        μ = median(gdf.avg_model_loss)
        (; loss = μ)
    end
    @info loss_stats
    loss_stats = Dict(zip(loss_stats.tokenizer, loss_stats.loss))

    markers = [:x, :+]
    for (idx, tokenizer) in enumerate(plt_tokenizers)
        tokenizer ∉ keys(loss_stats) && continue
        tokenizer ∉ keys(usage_stats) && continue
        stats = BSON.load(usage_stats[tokenizer])
        fertility_counts = :val ∈ keys(stats) ? stats[:val][:fertility] : stats[:fertility]
        μ, σ = mean_and_std(collect(keys(fertility_counts)), Weights(collect(values(fertility_counts))))
        loss = -loss_stats[tokenizer]
        @info tokenizer loss μ σ

        # pretty print token count
        token_count = stats[:tokenizer][:vocab_size]
        if token_count % 1000 == 0
            token_count = "$(fld(token_count, 1000))k"
        else
            token_count = "$(token_count)"
        end

        scatter!(ax, [μ], [loss];
            label=TOKENIZERS[tokenizer] * " ($token_count)",
            marker=@show markers[idx % length(markers) + 1]
        )
    end

    axislegend(ax, position=:lt, orientation=:vertical)

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

    tokenizers = OrderedDict(
        "smirk" => "smirk (ours)",
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
    )
    f = Figure(;
        size=72 .* (7, 2),
        figure_padding=(1, 1, 1, 5),
    )
    ax = Axis(f[1,1];
        ylabel="Out of Vocab Rate",
        limits=(nothing, (0, 100)),
        ytickformat="{:.0f}%",
        xticklabelrotation = 0.4,
        xticks=(1:length(tokenizers), collect(values(tokenizers))),
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
    for (idx, (tok_name, plt_name)) in enumerate(tokenizers)
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

function figure_ngram_fits()
    df = TokenizerStats.ngram_perf()
    plt_toks = [
        "smirk",
        "ibm/MoLFormer-XL-both-10pct-oov",
        "devalab/molgpt-moses",
        "rxn4chemistry/rxn_yields",
        "rxn4chemistry/rxnfp",
        "SmilesPE/SPE_ChEMBL",
        "MolecularAI/Chemformer",
        "seyonec/ChemBERTa-zinc-base-v1",
        "meta-llama/Meta-Llama-3.1-8B",
        "Xenova/gpt-4o",
        "google/gemma-7b",
        "character",
    ]
    df = subset(df,
        :dataset => ByRow(==("realspace_v4_dev")),
        :tokenizer => ByRow(x -> x in plt_toks)
    )
    pos = Int[]
    group = Int[]
    log_prob = Float64[]
    df.tokenizer = categorical(df.tokenizer; levels=plt_toks)

    ngrams = levels(df.ngram)
    tokenizer = levels(df.tokenizer)

    f = Figure(; size=72 .* (7, 2))
    ax = Axis(f[1,1];
        limits=(nothing, (1, nothing)), xlabel="n-gram size",
        ylabel="KL-Divergence [nats]",
        xticks = (1:length(tokenizer), map(n -> TOKENIZERS[n], tokenizer)),
        yticks = [1, 10, 100],
        xticklabelrotation = 0.4,
        xticksvisible=false,
        yscale=log10,
        xgridvisible=false,
        xminorgridvisible=false,
        yminorticks = IntervalsBetween(10),
        yminorticksvisible=true,
        yminorgridvisible=true,
    )
    h = barplot!(ax, levelcode.(df.tokenizer), -1 .* df.log_odds;
        dodge=df.ngram,
        color=df.ngram,
        colormap=:Set2_5,
    )

    ds_elements = map(1:length(ngrams)) do gdx
        PolyElement(polycolor=gdx,
            colormap=h.colormap,
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
    tok_log_prob!(f[2,2], cb, path("rxn4chemistry/rxn_yields")..., smi; direction)
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

function figure_ngram_info_loss()
    f = Figure(size=72 .* (4.5, 3))
    cb = Colorbar(f[1:2, 4];
        label="Log Odds Ratio",
        colormap=:vik,
        colorrange=(-17, 17),
        tellheight=true,
    )

    smi = "C(=Cc1ccccc1)C1=[O+][Cu-3]2([O+]=C(C=Cc3ccccc3)CC(c3ccccc3)=[O+]2)[O+]=C(c2ccccc2)C1"

    # Load model
    ref_file = joinpath(@__DIR__, "stats", "character", "realspace_v4_dev.bson")
    ngram, ref_tok, ref_info = TokenizerStats.load_ngram_model(ref_file)
    ref_code = pyconvert(Vector{Int}, ref_tok(smi)["input_ids"])
    kwargs = (; ngram, ref_tok, ref_code)

    tok_info_loss!(f[1,1], cb, "smirk", smi; kwargs...)
    tok_info_loss!(f[1,2], cb, "ibm/MoLFormer-XL-both-10pct-oov", smi; kwargs...)
    tok_info_loss!(f[1,3], cb, "SmilesPE/SPE_ChEMBL", smi; kwargs...)
    tok_info_loss!(f[2,1], cb, "MolecularAI/Chemformer", smi; kwargs...)
    tok_info_loss!(f[2,2], cb, "devalab/molgpt-moses", smi; kwargs...)
    tok_info_loss!(f[2,3], cb, "rxn4chemistry/rxn_yields", smi; kwargs...)

    # Format plot
    Label(f[0, :], smi, fontsize=6)
    Label(f[end+1, :], "Predicted Tokens", fontsize=8)
    resize_to_layout!(f)
    colgap!(f.layout, 3)
    rowgap!(f.layout, 3)

    return f
end

function box_token!(ax, token_id::Int, code_pos::Int; kwargs...)
    token_id += 1
    point = [
        (token_id, code_pos),
        (token_id, code_pos + 1),
        (token_id + 1, code_pos + 1),
        (token_id + 1, code_pos),
        (token_id, code_pos),
    ]
    point = map(x -> (x[1] - 0.5, x[2] - 0.5), point)
    lines!(ax, point; color=:red, linewidth=0.2, kwargs...)
end

function tok_info_loss!(f, cb, tok::String, smi::String; ngram, ref_tok, ref_code)

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
    # @assert any(masked) == true "expected at least one token to be masked"
    @assert length(masked) == length(ref_code)
    i, P, Q = TokenizerStats.information_loss(ngram, ref_code, masked; N=2)

    # Remove special tokens
    vocab = TokenizerStats.nonspecial_vocab(ngram)
    P = P[vocab .+ 1, :]
    Q = Q[vocab .+ 1, :]

    odds_ratio = @. (Q - log(1 - exp(Q))) - (P - log(1 - exp(P)))
    @info "masked for $name" masked_tokens=join(ref_tokens[masked], " ") extrema(odds_ratio)

    ax = Axis(f;
        title = "$name: $(round(i; sigdigits=3))",
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

    # # Highlight the correct token
    # for (code_pos, token_id) in enumerate(ref_code)
    #     box_token!(ax, token_id, code_pos; linewidth=0.5, color=:black)
    # end

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

function figure_kl_v_info_loss()
    model_loss = TokenizerStats.model_loss_stats()
    info_loss = TokenizerStats.avg_molnet_info_loss()

    model_loss = combine(groupby(model_loss, [:tokenizer, :split, :ngram])) do gdf
        return (;
            avg_model_loss = median(gdf.avg_model_loss),
            max_model_loss = maximum(gdf.max_model_loss),
            vocab_size = first(gdf.vocab_size),
            samples = sum(gdf.samples),
        )
    end
    select!(model_loss, Not([:samples]))
    select!(info_loss, Not([:samples, :vocab_size]))
    df = leftjoin(model_loss, info_loss, on=[:tokenizer, :split, :ngram])
    plt_tokenizers = [
        "smirk" => :star5,
        "ibm/MoLFormer-XL-both-10pct-oov" => :diamond,
        "devalab/molgpt-moses" => :rect,
        "rxn4chemistry/rxn_yields" => :cross,
        "MolecularAI/Chemformer" => :utriangle,
        "SmilesPE/SPE_ChEMBL" => :x,
    ]
    subset!(df,
        :tokenizer => ByRow(x -> x in first.(plt_tokenizers)),
        :ngram => ByRow(>(1)),
        :split => ByRow(==(:val)),
    )
    df.tokenizer = categorical(df.tokenizer)

    display(df[!, [:tokenizer, :ngram, :avg_model_loss, :avg_info_loss]])

    f = Figure(; size=72 .* (4.5, 3))
    ax = Axis(f[1,1];
        xlabel="KL-Divergence [nats]",
        ylabel="Entropy of Unknowns [nats]",
        yticks=[0, 0.05, 0.25, 1, 4],
        yscale=Asinh(0.05),
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(4),
        yminorgridvisible=true,
        xgridvisible=false,
    )
    h = scatter!(ax, -df.avg_model_loss, df.avg_info_loss;
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
        nbanks=2,
        halign=:right,
        valign=:top,
    )



    return f
end
