struct SlidingWindow{N, T}
    data::Vector{T}
end

SlidingWindow(data::Vector{T}, n::Int) where {T} = SlidingWindow{n, T}(data)

function Base.iterate(sw::SlidingWindow{N}, state=1) where {N}
    if state + N - 1 > length(sw.data)
        return nothing
    end
    window = view(sw.data, state:state + N - 1)
    return (tuple(window...), state + 1)
end

Base.length(sw::SlidingWindow{N}) where {N} = length(sw.data) - N + 1
Base.eltype(sw::SlidingWindow{N, T}) where {N, T} = NTuple{N, T}

function transition(bigrams::AbstractDict, vocab_size=0)
    I = first.(keys(bigrams))
    J = last.(keys(bigrams))
    V = collect(values(bigrams))
    vocab_size = vocab_size > 0 ? vocab_size : max(maximum(I), maximum(J))
    s = sparse(I .+ 1, J .+ 1, V, vocab_size + 1, vocab_size + 1)
    # s ./= sum(s; dims=2)

    w = vec(sum(s; dims=1)) .+ vec(sum(s; dims=2))
    p = sortperm(w; rev=true)
    permute!(s, p, p)
    return s
end

struct NGramModel{N, M}
    counts::Dict{NTuple{N, Int}, Int}
    priors::Dict{NTuple{M, Int}, Int}
    total::Int
    vocab_size::Int
end

token_ids(m::NGramModel) = range(0; length=m.vocab_size)

function NGramModel(dist::AbstractDict{<:Union{Int, NTuple}, Int}, vocab_size::Int)
    n = length(first(keys(dist)))
    K = NTuple{n, Int}
    grams = keytype(dist) <: NTuple ? keys(dist) : Iterators.map(tuple, keys(dist))
    counts = Dict{K, Int}(zip(grams, values(dist)))
    priors = Dict{NTuple{n-1, Int}, Int}()
    if n > 1
        for (gram, count) in counts
            pgram = condgram(gram)
            priors[pgram] = get(counts, gram, 0) + get(priors, pgram, 0)
        end
    end
    total = sum(values(dist); init=0) + vocab_size^n
    NGramModel(counts, priors, total, vocab_size)
end

condgram(gram::NTuple{N, Int}) where {N} = reverse(Base.tail(reverse(gram)))

function log_probability(model::NGramModel{1}, gram::NTuple{1, Int})
    count = get(model.counts, gram, 0) + 1
    return log(count) - log(model.total)
end

function log_probability(model::NGramModel{N}, gram::NTuple{N, Int}) where {N}
    next = get(model.counts, gram, 0) + 1
    prior = get(model.priors, condgram(gram), 0) + model.vocab_size^(N-1)
    return log(next) - log(prior)
end

function log_probability(m::NGramModel{N}, code::Vector{Int}) where {N}
    sum(Iterators.map(Base.Fix1(log_probability, m), SlidingWindow(code, N)); init=0.0)
end

