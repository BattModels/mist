using TokenizerStats: tokenizer_label, moments!, hist_nbins, find, tokenusage!
using GLMakie
using LinearAlgebra: normalize
using StatsBase: StatsBase, Histogram, fit, AbstractWeights
using JSON

function fit_hist(counts; kwargs...)
    x = map(Base.Fix1(parse, Int)∘string, collect(keys(counts)))
    w = Int.(collect(values(counts))) |> StatsBase.FrequencyWeights
    nbins = hist_nbins(:sturges, x, w)
    h = fit(Histogram, x, w; nbins, kwargs...)
    h = normalize(h)
    centers = StatsBase.midpoints(first(h.edges))
    return centers, h.weights
end

function collect_results()
    tokenizers = Dict{String, String}()
    for result in find(joinpath(@__DIR__, "stats"), r"stats-.*\.json")
        name = joinpath(splitpath(relpath(result, @__DIR__))[2:end-1])
        tokenizers[name] = result
    end
    return tokenizers
end

function figure_token_usage(results::Dict)
    f = Figure(; size=(950, 500))
    l = 1e-5
    ax = Axis(f[1,1];
        limits=((0, 100), (1e-10, 1)),
        xlabel="Token Relative Rank [%]",
        ylabel="Token Relative Usage [%]",
        yscale=log10,
        xtickformat="{:d}%",
    )

    for (name, file) in pairs(results)
        stats = JSON.parsefile(file)
        tokenusage!(ax, stats; label=name)
    end

    Legend(f[2,1], ax; tellwidth=true, tellheight=true, nbanks=3)
    return f
end

function figure_vocab_entropy_moments(results::Dict)
    f = Figure()
    ax = Axis(f[2,1];
        ylabel="Entropy Moments [bits]",
        limits=(nothing, (1e-1, nothing)),
        yscale=log10,
        xticklabelrotation = 0.4,
    )
    names = String[]
    xpos = Int[]
    heights = Float64[]
    dodge = Int[]
    for (idx, (name, file)) in enumerate(pairs(results))
        stats = JSON.parsefile(file)
        push!(names, name)
        for (i, m) in enumerate(stats["entropy"]["moments"])
            push!(xpos, idx)
            push!(heights, m)
            push!(dodge, i)
        end
    end
    colors = Makie.wong_colors()
    barplot!(ax, xpos, heights;
             dodge,
             color=colors[dodge],
             fillto=1e-4,
             )
    ax.xticks[] = (1:length(names), names)

    # Legend
    labels = ["Mean", "Variance", "Skewness", "Kurtosis"]
    elements = [PolyElement(polycolor = colors[i]) for i in 1:length(labels)]
    title = "Moments"
    Legend(f[1,1], elements, labels, title;
           tellwidth=false,
           orientation=:horizontal,
    )
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

function figure_oov_rate(results::Dict)
    f = Figure(size=(600, 300))
    ax = Axis(f[2,1];
              ylabel="Out Of Vocab Rate",
              limits=(nothing, (0, 100)),
              ytickformat="{:.0f}%",
              xticklabelrotation = 0.4,
    )

    ids = Int[]
    groups = Int[]
    names = String[]
    oov_rate = Float64[]
    for (idx, (name, file)) in enumerate(pairs(results))
        stats = JSON.parsefile(file)

        # Tabulate OOV Rate
        tok = TokenizerStats.load_tokenizer(name)
        oov_carbon = TokenizerStats.oov_rate(tok, TokenizerStats.carbon_tokens())
        oov_elements = TokenizerStats.oov_rate(tok, TokenizerStats.elements())
        oov = 100 .* (stats["out_of_vocab"] / stats["samples"], oov_carbon, oov_elements)

        push!(names, name)
        append!(ids, repeat([idx], length(oov)))
        append!(groups, 1:length(oov))
        append!(oov_rate, [oov...])
    end
    h = barplot!(ax, ids, oov_rate; dodge=groups, color=groups, gap=0.1)
    ax.xticks[] = (1:length(names), names)

    group_ids = 1:3
    @assert group_ids == unique(groups)
    elements = [ PolyElement(polycolor=i, colormap=h.colormap, colorrange=extrema(group_ids)) for i in unique(groups) ]

    Legend(f[1, 1], elements, ["Realspace", "Carbon", "Elements"], tellheight=true, tellwidth=false, orientation=:horizontal)

    resize_to_layout!(f)
    return f
end
