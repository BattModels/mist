@testitem "Info Loss" setup = [NGramModelSetup] begin
    using TokenizerStats: TokenizerStats, NGramModel, information_loss, fb_log_probability, cross_entropy_log
    using LogExpFunctions: logsumexp

    # Setup a random n-gram model
    N = 2
    vocab_size = 3
    m = NGramModel(randngram(N; vocab_size), vocab_size)
    @test length(m) == N
    code = rand(0:m.vocab_size-1, 10)
    mask = falses(length(code))

    @testset "no masked tokens" begin
        mask = falses(length(code))
        loss = information_loss(m, code, mask)
        @test loss isa Float64
        @test loss == 0
    end

    function check_information_loss(m, code, mask)
        code = TokenizerStats.MaskedCode(code, mask, -100)

        # Information loss
        loss = information_loss(m, code.code, code.mask)
        @test loss isa AbstractFloat
        @test loss >= 0 && isfinite(loss)

        # Reference information loss
        P_ref = fb_log_probability(m, code.code)
        @test P_ref isa Matrix{Float64}
        Q_ref = fb_log_probability(m, Vector(code); mask=code.mask_value)
        @test Q_ref isa Matrix{Float64}
        @test size(P_ref) == size(Q_ref) == (m.vocab_size, length(code))
        loss_expected = cross_entropy_log(vec(P_ref), vec(Q_ref))
        @test loss_expected isa AbstractFloat
        @test loss_expected >= 0 && isfinite(loss_expected)

        # Check that the loss is the same as the reference loss
        @test loss ≈ loss_expected atol = 1e-6

        # Check distributions are the same
        P = Vector{Float64}(undef, m.vocab_size)
        for idx in 1:length(code)
            TokenizerStats.fb_log_dist!(P, m, code.code, idx, N)
            @test P ≈ P_ref[:, idx] atol = 1e-6
            @test logsumexp(P) ≈ 0 atol = 1e-6

            TokenizerStats.fb_log_probability!(P, m, code; idx, N, mask=code.mask_value)
            @test P ≈ Q_ref[:, idx] atol = 1e-6
            @test logsumexp(P) ≈ 0 atol = 1e-6
        end
    end

    # Single Masked
    mask = falses(length(code))
    mask[fld(length(code), 2)] = true
    @testset "single masked" check_information_loss(m, code, mask)

    # Random mask 
    mask = rand(Bool, length(code))
    @testset "rand mask" check_information_loss(m, code, mask)
end

@testitem "fb_log_dist!" setup = [NGramModelSetup] begin
    using TokenizerStats: TokenizerStats, NGramModel, fb_log_dist!, token_ids, gram_odds
    using LogExpFunctions: logsumexp
    @testset "1-gram" begin
        N = 1
        vocab_size = 5
        m = NGramModel(randngram(N; vocab_size), vocab_size)
        P = Vector{Float64}(undef, m.vocab_size)
        code = rand(0:m.vocab_size-1, 2N - 1)

        fb_log_dist!(P, m, code, N, N)
        V = TokenizerStats.nonspecial_vocab_size(m)
        for (i, id) in enumerate(token_ids(m))
            count, marginal = gram_odds(m, (id,))
            P_ref = log1p(count) - log(marginal + V)
            @test P[i] ≈ P_ref atol = 1e-6
        end
    end

    @testset "$N-gram" for N in [1, 2, 3, 5]
        vocab_size = 5
        m = NGramModel(randngram(N; vocab_size), vocab_size)
        P = Vector{Float64}(undef, m.vocab_size)
        code = rand(0:m.vocab_size-1, 2N)

        P_ref = TokenizerStats.fb_log_probability(m, code; N)
        @test eltype(P_ref) == eltype(P)
        @test size(P_ref) == (length(P), length(code))
        @test all(isfinite, P_ref)
        @test all(isapprox(0; atol=1e-6), logsumexp(P_ref; dims=1))

        @testset "idx $idx" for idx in eachindex(code)
            fb_log_dist!(P, m, code, idx, N)
            @test all(isfinite, P)
            @test logsumexp(P) ≈ 0 atol = 1e-6
            @test P ≈ P_ref[:, idx] atol = 1e-6
        end
    end
end

@testitem "fb_log_probability!" setup = [NGramModelSetup] begin
    using LogExpFunctions: logsumexp
    using TokenizerStats: TokenizerStats, NGramModel, fb_log_probability!, MaskedCode
    @testset "1-gram" begin
        N = 1
        vocab_size = 3
        m = NGramModel(randngram(N; vocab_size), vocab_size)
        P = Vector{Float64}(undef, m.vocab_size)
        code = rand(1:m.vocab_size, 1)
        masked_code = MaskedCode(code, [true], -100)
        fb_log_probability!(P, m, masked_code; idx=1, N)
        @test all(isfinite, P)
        @test logsumexp(P) ≈ 0 atol = 1e-6

        # Test with mask set to false
        P_ref = copy(P)
        masked_code.mask .= false
        fb_log_probability!(P, m, masked_code; idx=1, N)
        @test all(isfinite, P)
        @test logsumexp(P) ≈ 0 atol = 1e-6
        @test P ≈ P_ref atol = 1e-6 # No change as collapses to unigram
    end

    @testset "$N-gram" for N = [2, 3, 5]
        vocab_size = 3
        m = NGramModel(randngram(N; vocab_size), vocab_size)
        P = Vector{Float64}(undef, m.vocab_size)
        code = rand(0:m.vocab_size-1, 2N)
        code = MaskedCode(code, rand(Bool, length(code)), -100)
        code = MaskedCode([1, 2, 3, 4, 5], [false, true, false, true, false], -100)

        P_ref = TokenizerStats.fb_log_probability(m, Vector(code); N, mask=code.mask_value)
        @test size(P_ref) == (length(P), length(code))
        @test all(isfinite, P_ref)
        @test all(isapprox(0; atol=1e-6), logsumexp(P_ref; dims=1))

        function check_counts(P, c, m, masked)
            @test all(isfinite, P)
            @test all(>=(0), P)
            @test sum(P) == m
            @test P == c
        end

        fc, fm, fmasked = TokenizerStats.forward_odds(m, Vector(code); N)
        bc, bm, bmasked = TokenizerStats.backward_odds(m, Vector(code); N)

        @testset "idx $idx" for idx in eachindex(code)
            fb_log_probability!(P, m, code; idx, N)
            @test all(isfinite, P)
            @test P ≈ P_ref[:, idx] atol = 1e-6
            @test logsumexp(P) ≈ 0 atol = 1e-6
        end
    end
end

@testitem "fb_log_probability! / fb_log_dist!" setup = [NGramModelSetup] begin
    using TokenizerStats: TokenizerStats, NGramModel, MaskedCode, fb_log_dist!, fb_log_probability!
    using LogExpFunctions: logsumexp

    N = 5
    vocab_size = 32
    m = NGramModel(randngram(N; vocab_size), vocab_size)
    P = Vector{Float64}(undef, m.vocab_size)
    P_ref = similar(P)
    code = rand(0:m.vocab_size-1, 2N)
    code = MaskedCode(code, falses(length(code)), -100)

    @testset "idx $idx" for idx in eachindex(code)
        fb_log_probability!(P, m, code; idx, N)
        @test all(isfinite, P)
        @test logsumexp(P) ≈ 0 atol = 1e-6

        fb_log_dist!(P_ref, m, code, idx, N)
        @test all(isfinite, P_ref)
        @test logsumexp(P_ref) ≈ 0 atol = 1e-6

        @test P ≈ P_ref atol = 1e-6
    end
end

@testitem "forward/backward counts!" setup = [NGramModelSetup] begin
    using TokenizerStats: TokenizerStats, NGramModel, MaskedCode, forward_counts!, backward_counts!, forward_odds, backward_odds, condgram, condgram_backward
    N = 5
    vocab_size = 3
    m = NGramModel(randngram(N; vocab_size), vocab_size)
    P = Vector{Float64}(undef, m.vocab_size)

    mask = -100
    code = MaskedCode(rand(0:m.vocab_size-1, 2N), rand(2N) .> 0.9, mask)
    code = Vector(code)

    fc, fm, fmasked = forward_odds(m, code; mask, N)
    bc, bm, bmasked = backward_odds(m, code; mask, N)
    @test all(isfinite, fc) && all(isfinite, fm) && all(isfinite, fmasked)
    @test all(isfinite, bc) && all(isfinite, bm) && all(isfinite, bmasked)
    @test all(>=(0), fc) && all(>=(0), fm) && all(>=(0), fmasked)
    @test all(>=(0), bc) && all(>=(0), bm) && all(>=(0), bmasked)

    function check_counts(P, c, m, masked)
        @test all(isfinite, P)
        @test all(>=(0), P)
        @test sum(P) == m
        @test P == c
    end

    @testset "idx $idx" for idx in eachindex(code)
        fgram = condgram(code, idx, N)
        forward_counts!(P, m, fgram; mask)
        check_counts(P, fc[:, idx], fm[idx], fmasked[idx])

        bgram = condgram_backward(code, idx, N)
        backward_counts!(P, m, bgram; mask)
        check_counts(P, bc[:, idx], bm[idx], bmasked[idx])
    end

end
