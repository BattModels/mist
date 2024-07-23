# Estimate Number of Histogram Bins from data
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

function tokenizer_label(tok; vocab_size=false)
    name = tok.name
    if match(r"smirk-gpe", name) !== nothing
        name = "smirk-gpe"
    end
    if vocab_size
        name *= "\n$(tok.vocab_size) tokens"
    end
    return name
end

@recipe(TokenUsage, stats) do scene
    Attributes()
end

function collate_token_usage(ids::AbstractVector{<:Integer}, counts::Dict{<:AbstractString, Any})
    counts = Dict(( parse(Int, k) => v for (k, v) in pairs(counts) ))
    return collate_token_usage(ids, counts)
end
function collate_token_usage(ids::AbstractVector{T}, counts::Dict{T, <:Integer}) where {T}
    usage =  Vector{Int}(undef, length(ids))
    for (idx, token_id) in enumerate(ids)
        usage[idx] = get(counts, token_id, 0)
    end
    return usage
end

function Makie.plot!(plt::TokenUsage)
    stats = plt.stats[]
    vocab_size = Int(stats["tokenizer"]["vocab_size"])
    ids = 0:vocab_size-1
    usage = collate_token_usage(ids, stats["token_usage"])
    usage = usage ./ sum(usage)

    # Sort tokens by usage
    p = sortperm(usage; rev=true)
    usage = usage[p]
    ids = ids[p]

    x = range(0, 100; length=length(usage))
    stairs!(plt, x, usage)

end
