struct SlidingWindow{N,T}
    data::Vector{T}
end

SlidingWindow(data::Vector{T}, n::Int) where {T} = SlidingWindow{n,T}(data)

function Base.iterate(sw::SlidingWindow{N}, state=1) where {N}
    if state + N - 1 > length(sw.data)
        return nothing
    end
    window = view(sw.data, state:state+N-1)
    return (tuple(window...), state + 1)
end

Base.length(sw::SlidingWindow{N}) where {N} = length(sw.data) - N + 1
Base.eltype(::SlidingWindow{N,T}) where {N,T} = NTuple{N,T}

struct NGramModel{N,G}
    total::Int
    vocab_size::Int
    ngrams::G
    special_tokens::Vector{Int}
end

Base.length(::NGramModel{N}) where {N} = N

""" Iterator of all token ids (including special) for the model"""
token_ids(m::NGramModel) = range(0; length=m.vocab_size)

"""Vocab size, less special tokens"""
nonspecial_vocab_size(m::NGramModel) = length(token_ids(m)) - length(m.special_tokens)

nonspecial_vocab(m::NGramModel) = filter(∉(m.special_tokens), token_ids(m))

function NGramModel(tokenizer::Py, ngrams::Union{Tuple,Vector})
    vocab_size = pyconvert(Int, length(tokenizer))
    special_tokens = pyconvert(Vector{Int}, tokenizer.all_special_ids)

    # Don't remove in-use special tokens
    unk_token_id = pyconvert(Union{Nothing,Int}, tokenizer.unk_token_id)
    used_special = pyconvert(Vector{Int}, tokenizer("")["input_ids"])
    !isnothing(unk_token_id) && setdiff!(special_tokens, unk_token_id)
    setdiff!(special_tokens, used_special)

    return NGramModel(ngrams, vocab_size; special_tokens)
end

function NGramModel(ngrams::Union{Tuple,Vector}, vocab_size::Int; special_tokens::Vector{Int}=Int[])
    total = sum(values(first(ngrams)); init=0)
    ngrams = ntuple(i -> ngram_counts(ngrams[i], vocab_size), length(ngrams))
    N = length(ngrams)
    G = typeof(ngrams)
    return NGramModel{N,G}(total, vocab_size, ngrams, special_tokens)
end

function load_ngram_model(file::String, split=:train)
    # Open token stats file
    suffix = last(splitext(file))
    if suffix == ".bson"
        ref = BSON.load(file)
    elseif suffix == ".jld"
        ref = deserialize(file)
    elseif suffix == ".jld2"
        stats = @timed jldopen(file, "r") do data
            NamedTuple([:tokenizer => data["tokenizer"], split => data[string(split)]])
        end
        ref = stats.value
        @debug "loaded ngram model" file stats.time stats.bytes stats.gctime stats.gcstats.total_time stats.compile_time stats.recompile_time
    else
        error("unknown filetype: $file")
    end

    # Extract tokenizer info
    name = ref[:tokenizer][:name]
    name = startswith(name, "smirk-gpe") ? "./" * name : name
    tok = load_tokenizer(name)
    vocab_size = pyconvert(Int, length(tok))
    ngram = NGramModel(tok, ref[split][:ngrams])
    info = (;
        name=ref[:tokenizer][:name],
        unk_token_id=ref[:tokenizer][:unk_token_id],
        vocab_size,
        split,
        sha256=bytes2hex(open(SHA.sha256, file))
    )
    return ngram, tok, info
end

function ngram_counts(dist::AbstractDict{<:Union{<:Integer,NTuple},<:Union{Integer,Float32}}, vocab_size::Int)
    n = length(first(keys(dist)))
    K = NTuple{n,Int}
    V = valtype(dist)
    grams = keytype(dist) <: NTuple ? keys(dist) : Iterators.map(tuple, keys(dist))
    for gram in grams
        @assert all(t -> 0 <= t < vocab_size, gram) "Expected all token ids to be ∈ [0, $vocab_size), got $gram"
    end
    keytype(dist) <: NTuple && return dist
    return Dict{K,V}(zip(grams, values(dist)))
end


""" Return the conditional n-gram `(..., x_i-1)` for the given n-gram `(..., x_i)`"""
condgram(gram::NTuple{N,Int}) where {N} = reverse(Base.tail(reverse(gram)))
condgram(code::AbstractVector{<:Integer}, edx::Int, length::Int) = ngram(code, edx - 1, length - 1)

""" Return the backward conditional n-gram `(x_i+1, ...)` for the given n-gram `(x_i, ...)`"""
condgram_backward(gram::NTuple{N,Int}) where {N} = Base.tail(gram)
condgram_backward(code::AbstractVector{<:Integer}, edx::Int, length::Int) = ngram(code, range(; start=edx + 1, length=length - 1))

""" Return the n-gram starting at `edx` of at most `length` """
ngram(code::AbstractVector{<:Integer}, edx::Int, length::Int) = ngram(code, range(; stop=edx, length))
function ngram(code::AbstractVector{<:Integer}, indices::UnitRange)
    indices = filter(in(eachindex(code)), indices)
    return ntuple(i -> code[indices[i]], length(indices))
end

struct MaskedCode{T,C<:AbstractVector{T},M<:AbstractVector{Bool}} <: AbstractVector{T}
    code::C
    mask::M
    mask_value::T
end
MaskedCode(code::Vector{T}, mask::Union{BitVector,Vector{Bool}}, mask_value::T) where {T} = MaskedCode{T}(code, BitVector(mask), mask_value)
MaskedCode{T}(code::C, mask::M, mask_value) where {T,C<:AbstractVector{T},M<:AbstractVector{Bool}} = MaskedCode{T,C,M}(code, mask, mask_value)

Base.getindex(mc::MaskedCode, i::Int) = mc.mask[i] ? mc.mask_value : mc.code[i]
Base.eltype(::MaskedCode{T}) where {T} = T
Base.length(mc::MaskedCode) = length(mc.code)
Base.view(mc::MaskedCode{T}, indices...) where {T} = MaskedCode(view(mc.code, indices...), view(mc.mask, indices...), mc.mask_value)
Base.size(mc::MaskedCode) = size(mc.code)

"""
    c, m = gram_odds(m::NGramModel, gram::NTuple{N, Int})

The observed `c` and marginal `m` counts for the n-gram `gram`.
No smoothing is applied
"""
function gram_odds(model::NGramModel, gram::NTuple{N,<:Integer}) where {N}
    @assert 1 <= N <= length(model)
    counts = get(model.ngrams[N], gram, 0)
    marginal = N == 1 ? model.total : get(model.ngrams[N-1], condgram(gram), 0)
    return counts, marginal
end

""" Computes the log-probability of `gram` using the N-gram `model` with Add-1 Smoothing """
function log_probability(model::NGramModel, gram::NTuple{N,<:Integer}) where {N}
    c, n = gram_odds(model, gram)
    ln_c = any(∈(model.special_tokens), gram) ? log(c) : log1p(c)
    ln_n = log_smoothed_counts(n, 1, nonspecial_vocab_size(model))
    return ln_c - ln_n
end

""" Computes the log-odds of `gram` using the N-gram `model` with Add-1 Smoothing """
function log_odds(model::NGramModel, gram::NTuple{N,<:Integer}) where {N}
    c, n = gram_odds(model, gram)
    ln_c = any(∈(model.special_tokens), gram) ? log(c) : log1p(c)
    return ln_c - log_smoothed_counts(n - c, 1, nonspecial_vocab_size(model))
end

"""
Computes the cross-entropy loss using an `N`-gram model for `code`.
"""
function cross_entropy(model::NGramModel, code::Vector{<:Integer}; N=length(model))
    @assert !any(∈(model.special_tokens), code)
    loss = 0.0
    V = nonspecial_vocab_size(model)
    for i in 1:length(code)
        c, m = gram_odds(model, ngram(code, i, N))
        loss += log1p(c) - log(m + V)
    end
    return -loss
end

"""
Computes the cross entropy loss of the code `code` given the log-probabilities `ℓ`.
Will ignore tokens with id `ignore_id` in `code`
"""
function cross_entropy(ℓ::AbstractMatrix, code::Vector{<:Integer}; ignore_id=-100)
    H = 0.0
    for i in axes(ℓ, 2)
        code[i] == ignore_id && continue
        H += ℓ[code[i]+1, i]
    end
    return -H
end

"""
Computes the KL-Divergence `∑ p * log(p/q)` between `p = exp(P)` and `q = exp(Q)`
(P & Q are log-probabilities) using Kahan-Babuska-Neumaier summation
"""
function kl_divergence(P::AbstractArray, Q::AbstractArray)
    H = zero(eltype(P))
    c = zero(H)
    @assert size(P) == size(Q)
    for (p, q) in zip(P, Q)
        i = xexpy(p - q, p)
        t = H + i
        if abs(H) >= abs(i)
            c += (H - t) + i
        else
            c += (i - t) + H
        end
        H = t
    end
    return H + c
end

"""
Computes the log-probability `ℓ` of `code` using the N-gram `model`. Returns a
`Matrix{Float64}` of size `(model.vocab_size, length(code))`
"""
function log_probability(model::NGramModel, code::Vector; N=length(model))
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
                ell[j, i] = log1p(c) - log(m + V)
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
function fb_log_probability(m::NGramModel, code::AbstractVector{<:Integer}; mask::Integer=-100, N=length(m))
    fc, fm, fmasked = forward_odds(m, code; mask, N)
    bc, bm, bmasked = backward_odds(m, code; mask, N)

    # Compute forward and backward probabilities
    V = nonspecial_vocab_size(m)
    f_prob = @. log_smoothed_counts(fc, fmasked', V) - log_smoothed_counts(fm, fmasked .+ 1, V)'
    b_prob = @. log_smoothed_counts(bc, bmasked', V) - log_smoothed_counts(bm, bmasked .+ 1, V)'

    # If forward/backward collapses to the unigram distribution, don't double count
    for idx in eachindex(code)
        f_collapse = isempty(condgram(code, idx, N))
        b_collapse = isempty(condgram_backward(code, idx, N))
        if f_collapse && b_collapse
            fill_unigram_dist!(f_prob[:, idx], m)
        elseif f_collapse
            f_prob[:, idx] .= b_prob[:, idx]
        elseif !b_collapse
            @. f_prob[:, idx] = f_prob[:, idx] + b_prob[:, idx]
        end
    end

    # Compute joint probability
    marginal = logsumexp(f_prob; dims=1)
    return f_prob .- marginal
end

function fb_log_probability!(Pf::Vector{Float64}, m::NGramModel, code::AbstractVector{<:Integer}; idx::Integer=1, N::Integer, mask::Integer=-100)
    fgram = condgram(code, idx, N)
    bgram = condgram_backward(code, idx, N)

    # Count matching n-grams
    Pb = similar(Pf)
    forward_counts!(Pf, m, fgram; mask)
    backward_counts!(Pb, m, bgram; mask)

    # Smooth forward/backward distributions
    V = nonspecial_vocab_size(m)
    fmasked = count(==(mask), fgram)
    fmarginal = sum(Pf)
    @. Pf = log_smoothed_counts(Pf, fmasked, V) - log_smoothed_counts(fmarginal, fmasked + 1, V)
    bmasked = count(==(mask), bgram)
    bmarginal = log_smoothed_counts(sum(Pb), bmasked + 1, V)
    @. Pb = log_smoothed_counts(Pb, bmasked, V) - bmarginal

    # Compute Joint probability
    if isempty(fgram)
        Pf .= Pb
    elseif !isempty(bgram)
        @. Pf = Pf + Pb
    end

    marginal = logsumexp(Pf)
    Pf .-= marginal

    return nothing
end

"""
Computes `log(counts + vocab_size^nmasked)` in a numerically stable way
"""
function log_smoothed_counts(counts::Real, nmasked::Integer, vocab_size::Integer)
    ln_mask = nmasked * log(vocab_size)
    ln_counts = log(counts)
    if counts == 0
        return ln_mask
    elseif ln_counts > ln_mask
        return ln_counts + log1pexp(ln_mask - ln_counts)
    else
        return ln_mask + log1pexp(ln_counts - ln_mask)
    end
end

function forward_odds(m::NGramModel, code::AbstractVector; mask::Integer=-100, N=length(m))
    ctype = valtype(m.ngrams[N])
    ctype = ctype isa Integer ? UInt64 : Float32
    counts = Matrix{ctype}(undef, m.vocab_size, length(code))
    marginal = zeros(Int, length(code))
    n_masked = Vector{Int}(undef, length(code))
    Threads.@threads for i in 1:length(code)
        cgram = condgram(code, i, N)
        # cgram = lstrip(cgram, mask)
        n_masked[i] = count(==(mask), cgram)
        token_marginal = zero(ctype)
        for (j, token) in enumerate(token_ids(m))
            c = masked_counts(m, (cgram..., token), mask)
            counts[j, i] = c
            token_marginal += c
        end
        marginal[i] = token_marginal
    end
    return counts, marginal, n_masked
end

function forward_counts!(P::Vector{T}, m::NGramModel, cgram::NTuple{N,Int}; mask::Integer=-100) where {T,N}
    if isempty(cgram)
        fill_unigram_dist!(P, m)
        return P
    end
    fill!(P, 0)

    int_max = T isa AbstractFloat ? maxintfloat(T) : typemax(T)
    @assert 0 < N < length(m)
    for (canidate, count) in m.ngrams[N+1]
        if masked_match(cgram, canidate; mask)
            i = last(canidate) + 1
            P[i] += count
            @assert P[i] < int_max
        end
    end
    return P
end

function backward_odds(m::NGramModel, code::AbstractVector; mask::Integer=-100, N=length(m))
    code = reverse(code)
    ctype = valtype(m.ngrams[N])
    ctype = ctype isa Integer ? UInt64 : Float32
    counts = Matrix{ctype}(undef, m.vocab_size, length(code))
    marginal = zeros(Int, length(code))
    n_masked = Vector{Int}(undef, length(code))
    Threads.@threads for (i, ri) in collect(enumerate(range(length(code), 1; step=-1)))
        cgram = reverse(condgram(code, i, N))
        # cgram = rstrip(cgram, mask)
        n_masked[ri] = count(==(mask), cgram)
        token_marginal = zero(ctype)
        for (j, token) in enumerate(token_ids(m))
            c = masked_counts(m, (token, cgram...), mask)
            counts[j, ri] = c
            token_marginal += c
        end
        marginal[ri] = token_marginal
    end
    return counts, marginal, n_masked
end

function backward_counts!(P::Vector{T}, m::NGramModel, cgram::NTuple{N,Int}; mask::Integer=-100) where {T,N}
    if isempty(cgram)
        fill_unigram_dist!(P, m)
        return P
    end
    fill!(P, 0)

    int_max = T isa AbstractFloat ? maxintfloat(T) : typemax(T)
    @assert 0 < N < length(m)
    for (canidate, count) in m.ngrams[N+1]
        if masked_match(reverse(cgram), reverse(canidate); mask)
            i = first(canidate) + 1
            P[i] += count
            @assert P[i] < int_max
        end
    end
    return P
end

"""
    gram = lstrip(gram, mask)

Remove mask tokesn from the left or right side of the gram
"""
function Base.lstrip(gram::NTuple{N,<:Integer}, mask::Integer) where {N}
    for i in 1:N
        gram[i] == mask || return gram[i:end]
    end
    return tuple()
end

"""
    gram = rstrip(gram, mask)

Remove mask tokesn from the left or right side of the gram
"""
function Base.rstrip(gram::NTuple{N,<:Integer}, mask::Integer) where {N}
    for i in range(N, 1; step=-1)
        gram[i] == mask || return gram[1:i]
    end
    return tuple()
end


"""
Computes the Information Loss (KL-Divergence) from masking out tokens in an input code for a
given n-gram model
"""
@annotate function information_loss(m::NGramModel, code::Vector{<:Integer}, mask::Union{BitVector,Vector{Bool}}; N=length(m))
    @assert 0 < N <= length(m)
    @assert length(code) == length(mask)
    loss = 0.0
    ctype = valtype(m.ngrams[N]) isa Integer ? UInt64 : Float64
    P = Vector{ctype}(undef, m.vocab_size)
    Q = similar(P)
    code = MaskedCode(code, mask, -100)
    for idx in eachindex(code)
        # If nothing is masked, the information
        # loss is zero
        sdx = max(idx - N + 1, firstindex(code))
        edx = min(idx + N - 1, lastindex(code))
        !any(mask[sdx:edx]) && continue

        # Compute P and Q distributions
        fb_log_dist!(P, m, code.code, idx, N)
        fb_log_probability!(Q, m, code; N, idx, mask=code.mask_value)

        # Update information loss
        loss += kl_divergence(P, Q)
    end
    return loss
end

function fill_unigram_dist!(P::AbstractVector, m::NGramModel)
    for (i, id) in enumerate(token_ids(m))
        P[i] = first(gram_odds(m, (id,)))
    end
    return P
end

function fb_log_dist!(Pf::Vector{Float64}, m::NGramModel, code::AbstractVector{<:Integer}, idx::Integer, N::Integer)
    Pb = similar(Pf)
    fgram = condgram(code, idx, N)
    bgram = condgram_backward(code, idx, N)
    V = nonspecial_vocab_size(m)
    for (idx, id) in enumerate(token_ids(m))
        fc = first(gram_odds(m, (fgram..., id)))
        Pf[idx] = fc
        bc = first(gram_odds(m, (id, bgram...)))
        Pb[idx] = bc
    end

    # Smooth distribution
    V = nonspecial_vocab_size(m)
    mf = sum(Pf)
    @. Pf = log_smoothed_counts(Pf, 0, V) - log_smoothed_counts(mf, 1, V)
    mb = sum(Pb)
    @. Pb = log_smoothed_counts(Pb, 0, V) - log_smoothed_counts(mb, 1, V)

    # Compute Joint probability
    if isempty(fgram)
        Pf .= Pb
    elseif !isempty(bgram)
        @. Pf = Pf + Pb
    end

    # Marginalize probability
    marginal = logsumexp(Pf)
    Pf .-= marginal

    return Pf
end

function masked_counts(m::NGramModel, gram::NTuple{N,<:Integer}, mask::Integer) where {N}
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

function masked_match(x::NTuple{Nx,<:Integer}, y::NTuple{Ny,<:Integer}; mask::Integer) where {Nx,Ny}
    for (a, b) in zip(x, y)
        if a == mask || b == mask
            continue
        elseif a != b
            return false
        end
    end
    return true
end


"""
Compute the information_loss from unknown tokens using a character-tokenizer as a reference
"""
@annotate function unk_information_loss(ngram::NGramModel, ref_tok::Py, tok::Py, encoding::Py; N=1:length(ngram), smi_column="smiles")
    unk_token_id = pyconvert(Int, tok.unk_token_id)
    code = pyconvert(Vector{Int}, encoding["input_ids"])
    (unk_token_id ∉ code) && return zeros(length(N))

    # Align both tokenizations
    smi_tokens = pyconvert(Vector{String}, tok.tokenize(encoding[smi_column]))
    ref_tokens = pyconvert(Vector{String}, ref_tok.tokenize(encoding[smi_column]))
    smi_tokens = rm_special_tokens(tok, smi_tokens)
    ref_tokens = rm_special_tokens(ref_tok, ref_tokens)
    A = align_unknown(ref_tokens, smi_tokens)

    # Mask out unknown tokens
    masked = map(!, vec(any(A; dims=2)))
    !(any(masked)) && return zeros(length(N)) # Unexpected, but possible if unk is from whitespace

    # Compute information_loss from unknown tokens
    ref_code = pyconvert(Vector{Int}, ref_tok(encoding[smi_column])["input_ids"])
    ref_code = rm_special_tokens(ref_tok, ref_code, length(masked))
    @assert length(masked) == length(ref_code)
    return map(n -> information_loss(ngram, ref_code, masked; N=n), N)
end

function rm_special_tokens(tok::Py, code::Vector{<:Integer}, n::Integer)
    # If the code is already of length n, return it
    length(code) == n && return code

    # Remove special tokens from the code
    unk_token_id = pyconvert(Int, tok.unk_token_id)
    special_tokens = pyconvert(Vector{Int}, tok.all_special_ids)
    special_tokens = setdiff(special_tokens, unk_token_id)
    return filter(∉(special_tokens), code)
end

function rm_special_tokens(tok::Py, tokens::Vector{String})
    if pyconvert(Bool, tok.__class__.__name__.startswith("T5Tokenizer"))
        tokens = filter(!isempty, replace.(tokens, "▁" => ""))
        @assert !isempty(tokens)
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
    i = (; idx=firstindex(a), char=1)
    j = (; idx=firstindex(b), char=1)
    I = Int[]
    J = Int[]
    is_unknown(x) = x == "[UNK]"
    unk_a_flag = false
    unk_b_flag = false
    mark = nothing
    failed = false
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
                i = (; idx=i.idx + 1, char=1)
                j = _advance_idx(b[j.idx], j)
            end
            if unk_b
                i = _advance_idx(a[i.idx], i)
                j = (; idx=j.idx + 1, char=1)
            end
            unk_a_flag = unk_a
            unk_b_flag = unk_b

            # Mark start of unknown token stream
            mark = (; i, j, unk_a_flag, unk_b_flag, n=length(I))
        elseif ac == bc
            # Tokens are aligned, emit alignment entry
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
                failed = true
                break
            end
        else
            failed = true
            break
        end
    end

    if failed
        @error "Alignment failed" a b sparse(I, J, trues(length(I)), length(a), length(b))
        error("Failed to align unknown tokens")
    end

    return sparse(I, J, trues(length(I)), length(a), length(b))
end
