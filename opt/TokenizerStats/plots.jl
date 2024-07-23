using CairoMakie
using LinearAlgebra: normalize
using StatsBase: StatsBase, Histogram, fit, AbstractWeights
using JSON3

function collate_token_usage(ids, counts)
    usage =  Vector{Int}(undef, length(ids))
    for (idx, token_id) in enumerate(ids)
        usage[idx] = get(counts, token_id, 0)
    end
    return usage
end

hist_nbins(x::AbstractVector, w::AbstractWeights) = hist_nbins(:scott, x)
hist_nbins(method::Symbol, x::AbstractVector) = hist_nbins(method, x, StatsBase.UnitWeights{Int}(length(x)))

hist_nbins(method, x, w) = hist_nbins(Val(Symbol(method)), x, w)
hist_nbins(method::Symbol, x::AbstractVector, w::AbstractWeights) = hist_nbins(Val(method), x, w)
function hist_nbins(::Val{:scott}, x, w)
    σ = std(x, w)
    h = 3.5 * σ / cbrt(length(x))
    n = (maximum(x) - minimum(x)) / h
    return ceil(Int, n)
end
hist_nbins(::Val{:sqrt}, x, w) = ceil(Int, sqrt(length(x)))
hist_nbins(::Val{M}, args...) where {M} = MethodError(hist_nbins, M, args...)
hist_nbins(::Val{:sturges}, x, w) = ceil(Int, log2(length(x)) + 1)
hist_nbins(::Val{:sturges}, x, w::StatsBase.FrequencyWeights) = ceil(Int, log2(sum(w)) + 1)

function fit_hist(counts; kwargs...)
    x = map(Base.Fix1(parse, Int)∘string, collect(keys(counts)))
    w = Int.(collect(values(counts))) |> StatsBase.FrequencyWeights
    nbins = hist_nbins(:sturges, x, w)
    h = fit(Histogram, x, w; nbins, kwargs...)
    h = normalize(h)
    centers = StatsBase.midpoints(first(h.edges))
    return centers, h.weights
end

function find(dir, pattern)
    files = String[]
    for file in readdir(dir; join=true)
        if match(pattern, file) !== nothing
            push!(files, file)
        end
    end
    return files
end

function plot_token_usage!(plt, stats)
    vocab_size = Int(stats.tokenizer.vocab_size)
    ids = 0:vocab_size-1
    usage = collate_token_usage(ids, stats.token_usage)
    usage = usage ./ sum(usage)

    # Sort tokens by usage
    p = sortperm(usage; rev=true)
    usage = usage[p]
    ids = ids[p]

    x = range(0, 100; length=length(usage))
    stairs!(plt, x, usage; label=tokenier_label(stats.tokenizer))
end

function figure_token_usage()
    f = Figure()
    l = 1e-5
    ax = Axis(f[1,1];
        limits=((0, 100), (1e-10, 1)),
        xlabel="Token Relative Rank [%]",
        ylabel="Token Relative Usage [%]",
        yscale=log10,
        xtickformat="{:d}%",
    )

    for file in find(joinpath(@__DIR__, "..", "data"), r"stats-.*\.json")
        stats = JSON3.read(file)
        plot_token_usage!(ax, stats)
    end

    Legend(f[2,1], ax; tellwidth=false, tellheight=true)
    return f
end

@recipe(Moments, x, moments, extrema) do scene
    Attributes(
        sigma_level=1,
    )
end

function Makie.plot!(plt::Moments)
    mu = plt.moments[][1]
    var = plt.moments[][2]
    lower, upper = plt.extrema[]
    sigma_level = plt.sigma_level
    σ = @lift $sigma_level * sqrt(var)
    lines!(plt, float.([plt.x[], plt.x[]]), [lower, upper])
    crossbar!(plt, plt.x, mu, @lift($mu + $σ), @lift($mu - $σ))
    return plt
end

function tokenier_label(tok; vocab_size=false)
    name = tok.name
    if match(r"smirk-gpe", name) !== nothing
        name = "smirk-gpe"
    end
    if vocab_size
        name *= "\n$(tok.vocab_size) tokens"
    end
    return name
end

function figure_vocab_entropy()
    f = Figure()
    ax = Axis(f[2,1];
        ylabel="Entropy Moments [bits]",
        yscale=log10,
    )
    names = String[]
    xpos = Int[]
    heights = Float64[]
    dodge = Int[]
    for (idx, file) in enumerate(find(joinpath(@__DIR__, "..", "data"), r"stats-.*\.json"))
        stats = JSON3.read(file)
        push!(names, tokenier_label(stats.tokenizer; vocab_size=false))
        for (i, m) in enumerate(stats.entropy.moments)
            push!(xpos, idx)
            push!(heights, m)
            push!(dodge, i)
        end
    end
    colors = Makie.wong_colors()
    barplot!(ax, xpos, heights;
             dodge,
             color=colors[dodge],
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

    return f
end

function figure_fertility()
    f = Figure()
    ax = Axis(f[1,1];
        xlabel="Fertility",
        ylabel="Faction of Corpus [%]",
        limits=(nothing, (0, nothing)),
    )
    ax_unique = Axis(f[1,2];
        xlabel="Number of Unique Tokens",
        ylabel="Faction of Corpus [%]",
        limits=(nothing, (0, nothing)),
    )
    for file in find(joinpath(@__DIR__, "..", "data"), r"stats-.*\.json")
        stats = JSON3.read(file)
        name = tokenier_label(stats.tokenizer)
        x, y = fit_hist(stats.fertility)
        stairs!(ax, x, y; label=name)

        x, y = fit_hist(stats.nunique)
        stairs!(ax_unique, x, y; label=name)
    end
    Legend(f[2,:], ax; tellwidth=false, tellheight=true)


    return f
end
