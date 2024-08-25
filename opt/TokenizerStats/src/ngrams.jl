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

struct NGramModel{N, G}
    total::Int
    vocab_size::Int
    ngrams::G
    special_tokens::Vector{Int}
end

Base.length(m::NGramModel{N}) where {N} = N

""" Iterator of all token ids (including special) for the model"""
token_ids(m::NGramModel) = range(0; length=m.vocab_size)

"""Vocab size, less special tokens"""
nonspecial_vocab_size(m::NGramModel) = length(token_ids(m)) - length(m.special_tokens)

function NGramModel(tokenizer::Py, ngrams::Tuple)
    vocab_size = pyconvert(Int, length(tokenizer))
    special_tokens = pyconvert(Vector{Int}, tokenizer.all_special_ids)
    unk_token = pyconvert(Union{Nothing, String}, tokenizer.unk_token)
    !isnothing(unk_token) && setdiff!(special_tokens, unk_token)
    return NGramModel(ngrams, vocab_size; special_tokens)
end

function NGramModel(ngrams::Union{Tuple, Vector}, vocab_size::Int; special_tokens::Vector{Int}=Int[])
    total = sum(values(first(ngrams)); init=0)
    ngrams = ntuple(i -> ngram_counts(ngrams[i], vocab_size), length(ngrams))
    N = length(ngrams)
    G = typeof(ngrams)
    return NGramModel{N,G}(total, vocab_size, ngrams, special_tokens)
end

function load_ngram_model(file::String)
    ref = BSON.load(file)
    name = ref[:tokenizer][:name]
    tok = load_tokenizer(name)
    vocab_size = pyconvert(Int, length(tok))
    ngram = NGramModel(tok, ref[:ngrams])
    info =(;
        name=ref[:tokenizer][:name],
        unk_token_id=ref[:tokenizer][:unk_token_id],
        vocab_size,
        sha256=bytes2hex(SHA.sha256(read(file)))
    )
    return ngram, tok, info
end

function ngram_counts(dist::AbstractDict{<:Union{Int, NTuple}, Int}, vocab_size::Int)
    n = length(first(keys(dist)))
    K = NTuple{n, Int}
    grams = keytype(dist) <: NTuple ? keys(dist) : Iterators.map(tuple, keys(dist))
    for gram in grams
        @assert all(t -> 0 <= t < vocab_size, gram) "Expected all counts to be ∈ [0, vocab_size)"
    end
    return Dict{K, Int}(zip(grams, values(dist)))
end


""" Return the conditional n-gram `(..., x_i-1)` for the given n-gram `(..., x_i)`"""
condgram(gram::NTuple{N, Int}) where {N} = reverse(Base.tail(reverse(gram)))
condgram(code::Vector{Int}, edx::Int, length::Int) = ngram(code, edx-1, length-1)

""" Return the n-gram starting at `edx` of at most `length` """
ngram(code::Vector{Int}, edx::Int, length::Int) = ngram(code, range(; stop=edx, length))
function ngram(code::Vector{Int}, indices::UnitRange)
    indices = filter(in(eachindex(code)), indices)
    return ntuple(i -> code[indices[i]], length(indices))
end

"""
    c, m = gram_odds(m::NGramModel, gram::NTuple{N, Int})

The observed `c` and marginal `m` counts for the n-gram `gram`.
No smoothing is applied
"""
function gram_odds(model::NGramModel, gram::NTuple{N, Int}) where {N}
    @assert 1 <= N <= length(model)
    counts = get(model.ngrams[N], gram, 0)
    marginal = N == 1 ? model.total : get(model.ngrams[N-1], condgram(gram), 0)
    return counts, marginal
end

function log_probability(model::NGramModel, gram::NTuple{N, Int}) where {N}
    c, n = gram_odds(model, gram)
    ln_c = any(∈(model.special_tokens), gram) ? log(c) : log1p(c)
    ln_n = log_smoothed_counts(n, 1, nonspecial_vocab_size(model))
    return ln_c - ln_n
end

function log_odds(model::NGramModel, gram::NTuple{N, Int}) where {N}
    c, n = gram_odds(model, gram)
    ln_c = any(∈(model.special_tokens), gram) ? log(c) : log1p(c)
    return ln_c - log_smoothed_counts(n - c, 1, nonspecial_vocab_size(model))
end

function autoregressive_kld(model::NGramModel, code::Vector{Int}; N=length(model))
    @assert !any(∈(model.special_tokens), code)
    counts = length(code)
    marginal = length(code) * nonspecial_vocab_size(model)
    for i in 1:length(code)
        c, m = gram_odds(model, ngram(code, i, N))
        counts += c
        marginal += m
    end
    return log(counts) - log(marginal)
end

function fb_log_probability(m::NGramModel, code::Vector; mask::Int=-100, N=length(m))
    fc, fm, fmasked = forward_odds(m, code; mask, N)
    bc, bm, bmasked = backward_odds(m, code; mask, N)

    # Total up forward/ backwards odds
    counts = fc .+ bc
    marginal = fm .+ bm
    masked = fmasked .+ bmasked
    @assert all(vec(sum(counts; dims=1)) .== marginal)

    # Add smoothed counts
    ell = Matrix{Float64}(undef, m.vocab_size, length(code))
    V = nonspecial_vocab_size(m)
    for i in 1:length(code)
        n_masked = masked[i]
        denom = log_smoothed_counts(marginal[i], n_masked + 1, V)
        for j in axes(ell, 1)
            if (j - 1) in m.special_tokens
                @assert counts[j, i] == 0
                ell[j, i] = -Inf
            else
                num = log_smoothed_counts(counts[j, i], n_masked, V)
                ell[j,i] = num - denom
            end
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
    marginal = zeros(Int, length(code))
    n_masked = Vector{Int}(undef, length(code))
    for i in 1:length(code)
        cgram = condgram(code, i, N)
        n_masked[i] = count(==(mask), cgram)
        for (j, token) in enumerate(token_ids(m))
            c = masked_counts(m, (cgram..., token), mask)
            counts[j,i] = c
            marginal[i] += c
        end
    end
    return counts, marginal, n_masked
end

function backward_odds(m::NGramModel, code::Vector; mask::Int=-100, N=length(m))
    code = reverse(code)
    counts = Matrix{UInt64}(undef, m.vocab_size, length(code))
    marginal = zeros(Int, length(code))
    n_masked = Vector{Int}(undef, length(code))
    for (i, ri) in enumerate(range(length(code), 1; step=-1))
        cgram = reverse(condgram(code, i, N))
        n_masked[ri] = count(==(mask), cgram)
        for (j, token) in enumerate(token_ids(m))
            c = masked_counts(m, (token, cgram...), mask)
            counts[j,ri] = c
            marginal[ri] += c
        end
    end
    return counts, marginal, n_masked
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
    P = fb_log_probability(m, code; N)
    Q = fb_log_probability(m, masked_code; N)
    for idx in eachindex(P)
        loss += exp(P[idx]) * (P[idx] - Q[idx])
    end
    return loss, P, Q
end

function masked_counts(m::NGramModel, gram::NTuple{N, Int}, mask::Int) where {N}
    N == 0 && return 0
    mask ∉ gram && return first(gram_odds(m, gram))
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

token_overlap(a, b) = token_overlap(tokens_and_offset(a), tokens_and_offset(b))

function tokens_and_offset(emb::Py)
    tokens = pyconvert(Vector{Int}, emb["input_ids"])
    offsets = pyconvert(Vector{NTuple{2, Int}}, emb["offset_mapping"])
    offsets = Iterators.map(o -> (o[1]+1, o[2]), offsets) # covert to closed intervals
    o = Iterators.filter(x -> x[2][2] >= x[2][1], zip(tokens, offsets))
    tokens = first.(o)
    offsets = last.(o)
    return (; tokens, offsets)
end

function token_overlap(a::NamedTuple, b::NamedTuple)
    overlap = Matrix{Bool}(undef, length(a.tokens), length(b.tokens))
    I = Int[]
    J = Int[]
    V = Bool[]
    for (i, ao) in enumerate(a.offsets)
        for (j, bo) in enumerate(b.offsets)
            a_less_b = ao[1] < bo[1] && ao[2] < bo[1]
            b_less_a = bo[1] < ao[1] && bo[2] < ao[1]
            if !a_less_b && !b_less_a
                push!(I, i)
                push!(J, j)
                push!(V, true)
            end
        end
    end
    return sparse(I, J, V), a.tokens, b.tokens
end

function token_overlap(text::String, tok_a::Py, tok_b::Py)
    emb_a = tok_a(text, return_offsets_mapping=true)
    emb_b = tok_b(text, return_offsets_mapping=true)
    return token_overlap(emb_a, emb_b)
end


