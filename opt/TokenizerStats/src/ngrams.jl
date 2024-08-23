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

Base.length(m::NGramModel{N}) where {N} = N
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
condgram(code::Vector{Int}, edx::Int, length::Int) = ngram(code, edx-1, length-1)

ngram(code::Vector{Int}, edx::Int, length::Int) = ngram(code, range(; stop=edx, length))
function ngram(code::Vector{Int}, indices::UnitRange)
    indices = filter(in(eachindex(code)), indices)
    return ntuple(i -> code[indices[i]], length(indices))
end

function gram_odds(model::NGramModel, gram::NTuple{N, Int}) where {N}
    @assert 1 <= N <= length(model)
    counts = get(model.ngrams[N], gram, 0) + 1
    if N == 1
        n = first(model.total) + model.vocab_size
        return counts, n
    end
    marginal = get(model.ngrams[N-1], condgram(gram), 0) + model.vocab_size
    return counts, marginal
end

function log_probability(model::NGramModel, gram::NTuple{N, Int}) where {N}
    c, n = gram_odds(model, gram)
    return log(c) - log(n)
end

function autoregressive_kld(model::NGramModel, code::Vector{Int}; N=length(model))
    counts = 0
    marginal = 0
    for i in 1:length(code)
        c, m = gram_odds(model, ngram(code, i, N))
        counts += c
        marginal += m
    end
    return log(counts) - log(marginal)
end

function fb_log_proability(m::NGramModel, code::Vector; mask::Int=-100, N=length(m))
    fc, fm, fmasked = forward_odds(m, code; mask, N)
    bc, bm, bmasked = backward_odds(m, code; mask, N)

    # Total up forward/ backwards odds
    counts = fc .+ bc
    marginal = fm .+ bm
    masked = fmasked .+ bmasked

    # Add marginal counts for first/last tokens (unigrams)
    marginal[1] += m.total[1]
    marginal[end] += m.total[1]

    # Add smoothed counts
    ell = Matrix{Float64}(undef, m.vocab_size, length(code))
    V = m.vocab_size
    for i in 1:length(code)
        n_masked = masked[i]
        denom = log_smoothed_counts(marginal[i], n_masked + 1, V)
        for j in 1:V
            num = log_smoothed_counts(counts[j, i], n_masked, V)
            ell[j,i] = num - denom
        end
    end
    return ell
end

"""
Computes `log(counts + vocab_size^nmasked)` in a numerically stable way
"""
function log_smoothed_counts(counts::Integer, nmasked::Int, vocab_size::Int)
    ln_mask = nmasked * log(vocab_size)
    ln_counts = log(counts)
    if counts == 0
        return ln_mask
    elseif ln_counts > ln_mask
        return ln_counts + log1p(exp(ln_mask - ln_counts))
    else
        return ln_mask + log1p(exp(ln_counts - ln_mask))
    end
end

function forward_odds(m::NGramModel, code::Vector; mask::Int=-100, N=length(m))
    counts = Matrix{UInt64}(undef, m.vocab_size, length(code))
    marginal = Vector{UInt64}(undef, length(code))
    masked_counts = Vector{Int}(undef, length(code))
    for i in 1:length(code)
        cgram = condgram(code, i, N)
        marginal[i] = masked_counts_unsmoothed(m, cgram, mask)
        masked_counts[i] = count(==(mask), cgram)
        for (j, token) in enumerate(token_ids(m))
            counts[j,i] = masked_counts_unsmoothed(m, (cgram..., token), mask)
        end
    end
    return counts, marginal, masked_counts
end

function backward_odds(m::NGramModel, code::Vector; mask::Int=-100, N=length(m))
    code = reverse(code)
    counts = Matrix{UInt64}(undef, m.vocab_size, length(code))
    marginal = Vector{UInt64}(undef, length(code))
    masked_counts = Vector{Int}(undef, length(code))
    for (i, ri) in enumerate(range(length(code), 1; step=-1))
        cgram = reverse(condgram(code, i, N))
        marginal[ri] = masked_counts_unsmoothed(m, cgram, mask)
        masked_counts[ri] = count(==(mask), cgram)
        for (j, token) in enumerate(token_ids(m))
            counts[j,ri] = masked_counts_unsmoothed(m, (token, cgram...), mask)
        end
    end
    return counts, marginal, masked_counts
end


"""
Computes the Information Loss (KL-Divergence) from masking out tokens in an input code for a
given n-gram model
"""
function information_loss(m::NGramModel, code::Vector{Int}, mask::Union{BitVector, Vector{Bool}}; N=length(m))
    @assert length(code) == length(mask)
    loss = 0.0
    mask_value = -100
    masked_code = copy(code)
    masked_code[mask] .= mask_value
    P = fb_log_proability(m, code; N)
    Q = fb_log_proability(m, masked_code; N)
    for idx in eachindex(P)
        loss += exp(P[idx]) * (P[idx] - Q[idx])
    end
    return loss, P, Q
end

function masked_counts_unsmoothed(m::NGramModel, gram::NTuple{N, Int}, mask::Int) where {N}
    # If gram is unmasked, likelihood is based on counts
    N == 0 && return 0
    mask ∉ gram && return first(gram_odds(m, gram)) - 1
    return masked_counts(m, gram, mask)
end

function masked_counts(m::NGramModel, gram::NTuple{N, Int}, mask::Int) where {N}
    @assert 1 <= N <= length(m)
    counts = 0
    for (candidate, c) in m.ngrams[N]
        if masked_match(gram, candidate; mask)
            counts += c
        end
    end
    return counts
end

function masked_match(x::NTuple{N, Int}, y::NTuple{N, Int}; mask::Int) where {N}
    for (a, b) in zip(x, y)
        if a == mask || b == mask
            continue
        elseif  a != b
            return false
        end
    end
    return true
end

function ngram_perf()
    rows = []
    stats_dir = joinpath(@__DIR__, "..", "stats")
    for file in find(stats_dir, r".*\.bson")
        data = BSON.load(file)
        file = relpath(file, stats_dir)
        tokenizer = joinpath(splitpath(file)[1:end-1])
        dataset = first(splitext(basename(file)))
        haskey(data, :ngram_log_odds) || continue
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

function information_loss(m::NGramModel, ref_tokenizer::Py, encoding::Py, unk_token_id::Int; kwargs...)
    code = pyconvert(Vector{Int}, encoding["input_ids"])
    offsets = pyconvert(Vector{NTuple{2, Int}}, encoding["offset_mapping"])
    ref_emb = ref_tokenizer(encoding["smiles"]; return_offsets_mapping=true)
    ref_offsets = pyconvert(Vector{NTuple{2, Int}}, ref_emb["offset_mapping"])
    ref_code = pyconvert(Vector{Int}, ref_emb["input_ids"])
    @assert all(x -> x[2] - x[1] == 1, ref_offsets)
    is_unknown = falses(length(ref_code))
    for (token, offset) in zip(code, offsets)
        if token == unk_token_id
            is_unknown[range(offset[1]+1, offset[2])] .= true
        end
    end
    return information_loss(m, ref_code, is_unknown; kwargs...)
end
