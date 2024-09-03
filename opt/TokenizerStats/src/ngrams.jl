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

nonspecial_vocab(m::NGramModel) = filter(∉(m.special_tokens), token_ids(m))

function NGramModel(tokenizer::Py, ngrams::Union{Tuple, Vector})
    vocab_size = pyconvert(Int, length(tokenizer))
    special_tokens = pyconvert(Vector{Int}, tokenizer.all_special_ids)

    # Don't remove in-use special tokens
    unk_token_id = pyconvert(Union{Nothing, Int}, tokenizer.unk_token_id)
    used_special = pyconvert(Vector{Int}, tokenizer("")["input_ids"])
    !isnothing(unk_token_id) && setdiff!(special_tokens, unk_token_id)
    setdiff!(special_tokens, used_special)

    return NGramModel(ngrams, vocab_size; special_tokens)
end

function NGramModel(ngrams::Union{Tuple, Vector}, vocab_size::Int; special_tokens::Vector{Int}=Int[])
    total = sum(values(first(ngrams)); init=0)
    ngrams = ntuple(i -> ngram_counts(ngrams[i], vocab_size), length(ngrams))
    N = length(ngrams)
    G = typeof(ngrams)
    return NGramModel{N,G}(total, vocab_size, ngrams, special_tokens)
end

function load_ngram_model(file::String, split=:train)
    ref = BSON.load(file)
    name = ref[:tokenizer][:name]
    name = startswith(name, "smirk-gpe") ? "./" * name : name
    tok = load_tokenizer(name)
    vocab_size = pyconvert(Int, length(tok))
    ngram = NGramModel(tok, ref[split][:ngrams])
    info =(;
        name=ref[:tokenizer][:name],
        unk_token_id=ref[:tokenizer][:unk_token_id],
        vocab_size,
        split,
        sha256=bytes2hex(SHA.sha256(read(file)))
    )
    return ngram, tok, info
end

function ngram_counts(dist::AbstractDict{<:Union{Int, NTuple}, Int}, vocab_size::Int)
    n = length(first(keys(dist)))
    K = NTuple{n, Int}
    grams = keytype(dist) <: NTuple ? keys(dist) : Iterators.map(tuple, keys(dist))
    for gram in grams
        @assert all(t -> 0 <= t < vocab_size, gram) "Expected all token ids to be ∈ [0, $vocab_size), got $gram"
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

"""
    loss = autoregressive_kld(model::NGramModel, code::Vector{Int}; N=length(model))

Computes the KL-Divergence loss (cross-entropy) using an `N`-gram model for `code`.
"""
function autoregressive_kld(model::NGramModel, code::Vector{Int}; N=length(model))
    @assert !any(∈(model.special_tokens), code)
    counts = 0
    marginal = 0
    loss = 0.0
    V = nonspecial_vocab_size(model)
    for i in 1:length(code)
        c, m = gram_odds(model, ngram(code, i, N))
        loss += log1p(c) - log_add(m, V)
    end
    return -loss
end

"""
    H = cross_entropy(ℓ::AbstractMatrix, code::Vector{Int}; ignore_id=-100)

Computes the cross entropy of the code `code` given the log-probabilities `ℓ`.
Will ignore tokens with id `ignore_id` in `code`
"""
function cross_entropy(ℓ::AbstractMatrix, code::Vector{Int}; ignore_id=-100)
    H = 0.0
    for i in axes(ℓ, 2)
        code[i] == ignore_id && continue
        H += ℓ[code[i] + 1, i]
    end
    return -H
end

function autoregressive_log_prob(model::NGramModel, code::Vector; N=length(model))
    ell = Matrix{Float64}(undef, model.vocab_size, length(code))
    V = nonspecial_vocab_size(model)
    for i in 1:length(code)
        cgram = condgram(code, i, N)
        for (j, token) in enumerate(token_ids(model))
            if (j - 1) in model.special_tokens
                ell[j, i] = -Inf
            else
                c, m = gram_odds(model, (cgram..., token))
                @assert (c + 1) <= (m + V)
                ell[j, i] = log1p(c) - log_add(m, V)
            end
        end
    end
    return ell
end

"""
    ℓ = fb_log_probability(m::NGramModel, code::Vector; mask::Int=-100, N=length(m))

Compute `ln P(x_i,j | x_{i-2}, x_{i-1}, x_{i+1}, x_{i+2})` for the given code `code` where
`i` is the index of the token in `code` and `j` is the index of the token in the vocabulary.
`N` is the n-gram order of the model, (i.e. `N=3` spans `x_{i-2}, x_{i-1}, x_{i+1}, x_{i+2}`)

Will marginalize over tokens with id `mask` in `code`. That is is `x_{i-2}` is masked,
then computes `P(x_i | x_{i-1}, x_{i+1}, x_{i+2})` instead.
"""
function fb_log_probability(m::NGramModel, code::Vector; mask::Int=-100, N=length(m))
    fc, fm, fmasked = forward_odds(m, code; mask, N)
    bc, bm, bmasked = backward_odds(m, code; mask, N)

    # Compute forward and backward probabilities
    V = nonspecial_vocab_size(m)
    f_prob = @. log1p(fc) - log_smoothed_counts(fm, fmasked, V)'
    b_prob = @. log1p(bc) - log_smoothed_counts(bm, bmasked, V)'

    # Compute joint probability
    ℓ = f_prob .+ b_prob
    marginal = log.(sum(exp, ℓ; dims=1))
    return ℓ .- marginal
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

""" Computes `log(x + y)` in a numerically stable way"""
log_add(x::Real, y::Real) = log_smoothed_counts(x, 1, y)

function forward_odds(m::NGramModel, code::Vector; mask::Int=-100, N=length(m))
    counts = Matrix{UInt64}(undef, m.vocab_size, length(code))
    marginal = zeros(Int, length(code))
    n_masked = Vector{Int}(undef, length(code))
    Threads.@threads for i in 1:length(code)
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
    Threads.@threads for (i, ri) in collect(enumerate(range(length(code), 1; step=-1)))
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
        !isfinite(P[idx])  && continue
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


"""
Compute the information_loss from unknown tokens using a character-tokenizer as a reference
"""
function  unk_information_loss(ngram::NGramModel, ref_tok::Py, tok::Py, encoding::Py; N=1:length(ngram))
    unk_token_id = pyconvert(Int, tok.unk_token_id)
    code = pyconvert(Vector{Int}, encoding["input_ids"])
    (unk_token_id ∉ code) && return zeros(length(N))

    # Align both tokenizations
    smi_tokens = pyconvert(Vector{String}, tok.tokenize(encoding["smiles"]))
    ref_tokens = pyconvert(Vector{String}, ref_tok.tokenize(encoding["smiles"]))
    smi_tokens = rm_special_tokens(tok, smi_tokens)
    ref_tok = rm_special_tokens(ref_tok, ref_tokens)
    A = align_unknown(ref_tokens, smi_tokens)

    # Compute information_loss from unknown tokens
    masked = map(!, vec(any(A; dims=2)))
    @info "masked: $masked"
    @assert any(masked) == true "expected at least one token to be masked, eval: $smi_tokens, ref: $ref_tokens, smi: $(join(ref_tokens, "")))"
    ref_code = pyconvert(Vector{Int}, ref_tok(encoding["smiles"])["input_ids"])
    ref_code = rm_special_tokens(ref_tok, ref_code, length(masked))
    @assert length(masked) == length(ref_code)
    return map(n -> first(information_loss(ngram, ref_code, masked; N=n)), N)
end

function rm_special_tokens(tok::Py, code::Vector{Int}, n::Int)
    # If the code is already of length n, return it
    length(code) == n  && return code

    # Remove special tokens from the code
    unk_token_id = pyconvert(Int, tok.unk_token_id)
    special_tokens = pyconvert(Vector{Int}, tok.all_special_ids)
    special_tokens = setdiff(special_tokens, unk_token_id)
    return filter(∉(special_tokens), code)
end

function rm_special_tokens(tok::Py, tokens::Vector{String})
    if pyconvert(Bool, tok.__class__.__name__.startswith("T5Tokenizer"))
        tokens = replace.(tokens, "▁" => "")
    end
    return tokens
end

function _advance_idx(token::String, index::NamedTuple)
    (; idx, char) = index
    if char < lastindex(token)
        char = nextind(token, char)
    else
        char = 1
        idx += 1
    end
    return (; idx, char)
end

"""
    A = align_unknown(a::Vector{String}, b::Vector{String})

Computes the alignment `A` between two tokenizations of the same input string.
The two tokenizations must be decode to the same string, barring deletions
from unknown tokens. i.e. normalizations must be applied to both.
"""
function align_unknown(a::Vector{String}, b::Vector{String})
    replace!(a, "<unk>" => "[UNK]")
    replace!(b, "<unk>" => "[UNK]")
    i = (; idx = firstindex(a), char = 1)
    j = (; idx = firstindex(b), char = 1)
    I = Int[]
    J = Int[]
    is_unknown(x) = x == "[UNK]"
    unk_a_flag = false
    unk_b_flag = false
    mark = nothing
    while i.idx <= lastindex(a) && j.idx <= lastindex(b)
        ac = a[i.idx][i.char]
        bc = b[j.idx][j.char]
        unk_a = is_unknown(a[i.idx])
        unk_b = is_unknown(b[j.idx])
        if xor(unk_a, unk_b)
            # Skip the unknown token, and the first char of the other
            # token stream, will keep skipping chars until steams are
            # realigned
            if unk_a
                i = (; idx=i.idx+1, char=1)
                j = _advance_idx(b[j.idx], j)
            end
            if unk_b
                i = _advance_idx(a[i.idx], i)
                j = (; idx=j.idx+1, char=1)
            end
            unk_a_flag = unk_a
            unk_b_flag = unk_b
        elseif ac == bc
            # Tokens are aligned, emit alignment entry
            mark = (; i, j, unk_a_flag, unk_b_flag, n=length(I))
            push!(I, i.idx)
            push!(J, j.idx)
            i = _advance_idx(a[i.idx], i)
            j = _advance_idx(b[j.idx], j)
            unk_a_flag = false
            unk_b_flag = false
        elseif ac != bc && (unk_a_flag || unk_b_flag)
            # Steams are still miss-aligned advance steam without an
            # active unknown token, until realigned
            if unk_b_flag
                i = _advance_idx(a[i.idx], i)
            end
            if unk_a_flag
                j = _advance_idx(b[j.idx], j)
            end
        elseif !isnothing(mark)
            if mark.unk_a_flag
                i = mark.i
            end
            if mark.unk_b_flag
                j = mark.j
            end
            unk_a_flag = mark.unk_a_flag
            unk_b_flag = mark.unk_b_flag
            I = I[1:mark.n]
            J = J[1:mark.n]
            new_mark = (; i, j, unk_a_flag, unk_b_flag, n=length(I))

            # Check that progress was made
            if new_mark != mark
                mark = new_mark
            else
                error("Failed to align unknown tokens: $a, $b")
            end
        else
            error("Failed to align: $a, $b")
        end
    end
    return sparse(I, J, trues(length(I)), length(a), length(b))
end

