@testitem "SlidingWindow" begin
    using TokenizerStats: SlidingWindow
    sw = SlidingWindow(collect(1:5), 2)
    @test eltype(sw) == NTuple{2, Int}
    @test length(sw) == 4
    @test collect(sw) == [(1, 2), (2,3), (3,4), (4,5)]
end

@testitem "log_smoothed_counts" begin
    using TokenizerStats: log_smoothed_counts
    @test log_smoothed_counts(1, 5, 256000) ≈ log(1 + BigInt(256000)^5) atol=1e-8
    @test log_smoothed_counts(5_000_000, 1, 256000) ≈ log(5_000_000 + BigInt(256000)^1) atol=1e-8
    @test log_smoothed_counts(0, 8, 256000) ≈ log(BigInt(256000)^8) atol=1e-8
    @test log_smoothed_counts(0, 0, 256000) == 0
end

@testsetup module NGramModelSetup
using TokenizerStats: NGramModel, SlidingWindow
using OnlineStats

export randngram

# Build random n-gram counts
function randngram(N=3; vocab_size=9)
    stats = map(n -> CountMap(NTuple{n, Int}), 1:N)
    corpus = rand(range(0; length=vocab_size), 8, 32)
    for i in axes(corpus, 2)
        for (n, s) in enumerate(stats)
            fit!(s, SlidingWindow(corpus[:, i], n))
        end
    end
    return map(value, stats)
end
end

@testitem "Construction" setup=[NGramModelSetup] begin
    using TokenizerStats: NGramModel, token_ids, gram_odds
    N = 4
    @testset "with special_tokens" begin
        m = NGramModel(randngram(N), 10; special_tokens=[9])
        @test m.total == 8 * 32
        @test length(m) == N
        @test TokenizerStats.nonspecial_vocab_size(m) == 9
        @test isempty(setdiff(token_ids(m), 0:9))
        c, m = gram_odds(m, (0,))
    end
    @testset "without special_tokens" begin
        m = NGramModel(randngram(N), 9)
        @test m.total == 8 * 32
        @test length(m) == N
        @test TokenizerStats.nonspecial_vocab_size(m) == 9
        @test isempty(setdiff(token_ids(m), 0:8))
        n, d = gram_odds(m, (0,))
        @test n isa Int && d isa Int
        @test 1 <= n <= m.total
        @test d == m.total
    end
end

@testitem "ngram/condgram" begin
    using TokenizerStats: condgram, ngram
    @test condgram((1, 2, 3)) == (1, 2)
    @test condgram([1, 2, 3, 4], 3, 2) == (2,)
    @test condgram([1, 2, 3, 4], 4, 2) == (3,)
    @test condgram([1, 2, 3, 4], 4, 3) == (2, 3)
    @test condgram([1, 2, 3, 4], 2, 3) == (1,)
    @test ngram(collect(1:4), 2, 3) == (1, 2)
    @test ngram(collect(1:4), 3, 3) == (1, 2, 3)
end

@testitem "gram_odds" setup=[NGramModelSetup] begin
    using TokenizerStats: NGramModel, gram_odds, token_ids, nonspecial_vocab_size, log_probability
    function check_odds(model)
        @testset "$N-gram" for N in 1:length(model)
            c = 0
            logprob = BigFloat(0)
            for token in token_ids(model)
                n, d = gram_odds(model, (token,))
                logprob += log_probability(model, (token,)) |> BigFloat |> exp
                @test d == model.total
                c += n
            end
            @test logprob ≈ 1 atol=1e-6
            @test c == model.total
        end
    end
    @testset "without special tokens" begin
        m = NGramModel(randngram(), 9)
        check_odds(m)
    end
    @testset "with special tokens" begin
        m = NGramModel(randngram(), 11; special_tokens=[9, 10])
        check_odds(m)
    end
end

@testitem "fb_log_probability" setup=[NGramModelSetup] begin
    using TokenizerStats: NGramModel, fb_log_probability
    function check(model, code)
        @testset "$N-gram" for N in 1:length(model)
            ℓ = fb_log_probability(model, code; N)
            @test all(<(0), ℓ)
            @test ℓ isa Matrix{Float64}
            @test size(ℓ, 1) == model.vocab_size
            @test size(ℓ, 2) == length(code)
            marginal = sum(exp, ℓ; dims=1, init=BigFloat(0))
            @test all(isapprox(1; atol=1e-6), marginal)
        end
    end
    @testset "without special tokens" begin
        m = NGramModel(randngram(), 9)
        check(m, [3, 4, 5, 8])
        check(m, [3, 4, -100, 5, 8])
    end
    @testset "with special tokens" begin
        m = NGramModel(randngram(), 11; special_tokens=[9, 10])
        check(m, [3, 4, 5, 8])
        check(m, [3, 4, -100, 5, 8])
        ℓ = fb_log_probability(m, [3, 4, -100, 8])
        special = m.special_tokens .- 1 # One Based Index, but Zero-based token ids
        all(ℓ[special, :] .== -Inf)
    end
end

@testitem "info loss" setup=[NGramModelSetup] begin
    using TokenizerStats: NGramModel, information_loss
    m = NGramModel(randngram(), 9)
    code = rand(1:8, 32)
    mask = code .== 4
    mask[1] = true
    i, P, Q = information_loss(m, code, mask)
    @test i isa Float64 && 0 < i
    @test P isa Matrix{Float64} && Q isa Matrix{Float64}
    @test size(P) == size(Q) == (m.vocab_size, length(code))

    @testset "P == Q" begin
        mask .= false
        i, P, Q = information_loss(m, code, mask)
        @test i == 0
        @test P == Q
    end
end

@testitem "autoregressive_kld" setup=[NGramModelSetup] begin
    using TokenizerStats: NGramModel, autoregressive_kld
    m = NGramModel(randngram(), 9)
    code = rand(1:8, 32)
    loss = zeros(length(m))
    for N in 1:length(m)
        loss[N] = autoregressive_kld(m, code; N)
    end
    @test all(<(0), loss)
end

@testitem "unk token information loss" setup=[NGramModelSetup] begin
    using PythonCall: pyconvert
    using TokenizerStats: NGramModel, load_tokenizer, unk_information_loss

    # Create a random reference ngram for the character tokenizer
    ref_tok = load_tokenizer("character")
    vocab_size = pyconvert(Int, length(ref_tok)) + 1 - length(ref_tok.all_special_tokens)
    ngram_counts = randngram(5; vocab_size)
    ngram = NGramModel(ref_tok, ngram_counts)

    # Use Molformer with a smiles that will have OOVs (rings)
    tok = load_tokenizer("ibm/MoLFormer-XL-both-10pct-oov")
    smi = "CC(=O)NCCC%01=CNC2=C%01C=C(C=C2)OC"
    emb = tok(smi)
    emb["smiles"] = smi
    loss = unk_information_loss(ngram, ref_tok, tok, emb)
    @test all(>=(0), loss) && all(isfinite, loss)
    @test loss isa Vector{Float64} && length(loss) == 5 == length(ngram)
end

@testitem "align_deletions" begin
    using TokenizerStats: align_unknown
    using SparseArrays: sparse, I

    function check_commutative(a, b)
        out = align_unknown(a, b)
        a == b && @test out == I
        @test out == align_unknown(b, a)'
        return out
    end

    @testset "single-char" begin
        check_commutative(["c", "a", "t"], ["c", "a", "t"])
        @test sparse([1, 2], [1, 2], ones(Int, 2), 3, 3) == check_commutative(["c", "a", "[UNK]"], ["c", "a", "t"])
        @test sparse([2, 3], [2, 3], ones(Int, 2), 3, 3) == check_commutative(["<unk>", "a", "t"], ["c", "a", "t"])
        out = check_commutative(["b", "a", "t", "t", "e", "r"], ["b", "a", "<unk>", "t", "e", "r"])
        @test sparse([1, 2, 4, 5, 6], [1, 2, 4, 5, 6], ones(Int, 5), 6, 6) ==  out
    end
    @testset "multi-char" begin
        check_commutative(["C", "Co", "C"], ["C", "Co", "C"])
        check_commutative(["C", "C", "o", "C"], ["C", "Co", "C"])
        out = check_commutative(["C", "C", "o", "C"], ["C", "<unk>", "C"])
        @test out == sparse([1, 4], [1, 3], trues(2), 4, 3)
    end
    @testset "double unknown" begin
        ref = ["C", "Z", "[", "C", "o", "+", "2", "]", "2", "c"]
        smi = ["C", "<unk>", "<unk>", "2", "c"]
        out = check_commutative(ref, smi)
        @test out == sparse([1, 9, 10], [1, 4, 5], trues(3), length(ref), length(smi))
    end
end
