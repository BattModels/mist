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

struct NGramModel{N, G}
    ngrams::G
    total::NTuple{N, Int}
    vocab_size::Int
end

token_ids(m::NGramModel) = range(0; length=m.vocab_size)

function ngram_counts(dist::AbstractDict{<:Union{Int, NTuple}, Int})
    n = length(first(keys(dist)))
    K = NTuple{n, Int}
    grams = keytype(dist) <: NTuple ? keys(dist) : Iterators.map(tuple, keys(dist))
    return Dict{K, Int}(zip(grams, values(dist)))
end

function NGramModel(ngrams, vocab_size::Int)
    ngrams = map(ngram_counts, ngrams)
    totals = map(g -> sum(values(g); init=0), ngrams)
    return NGramModel(ngrams, totals, vocab_size)
end

condgram(gram::NTuple{N, Int}) where {N} = reverse(Base.tail(reverse(gram)))

function log_probability(model::NGramModel, gram::NTuple{N, Int}) where {N}
    @assert N <= length(model.total)
    counts = get(model.ngrams[N], gram, 0) + 1
    if N == 1
        n = first(model.total) + model.vocab_size
        return log(counts) - log(n)
    end
    prior = get(model.ngrams[N-1], condgram(gram), 0) + model.vocab_size
    return log(counts) - log(prior)
end

log_probability(m::NGramModel{N}, code) where {N} = _ngram_log_odds(Val{N}(), m, code)
log_probability(m::NGramModel, code, N) = _ngram_log_odds(Val{N}(), m, code)
function _ngram_log_odds(::Val{N}, m::NGramModel, code::Vector{Int}) where {N}
    log_odds = 0.0
    for idx in range(1, min(N, length(code)))
        log_odds += log_probability(m, ntuple(i -> code[i], idx))
    end
    for gram in SlidingWindow(code, N)
        log_odds += log_probability(m, gram)
    end
    return log_odds
end

function ngram_perf()
    rows = []
    stats_dir = joinpath(@__DIR__, "..", "stats-2")
    for file in find(stats_dir, r".*\.bson")
        data = BSON.load(file)
        file = relpath(file, stats_dir)
        tokenizer = joinpath(splitpath(file)[1:end-1])
        dataset = first(splitext(basename(file)))
        push!(rows, (;
            tokenizer,
            dataset,
            vocab_size = data[:tokenizer][:vocab_size],
            samples = data[:samples],
            out_of_vocab = data[:out_of_vocab],
            unigram_log_odds = data[:ngram_log_odds][1],
            bigram_log_odds = data[:ngram_log_odds][2],
            trigram_log_odds = data[:ngram_log_odds][3],
            quadgram_log_odds = data[:ngram_log_odds][4],
            pentagram_log_odds = data[:ngram_log_odds][5],
        ))
    end
    return DataFrame(rows)
end

