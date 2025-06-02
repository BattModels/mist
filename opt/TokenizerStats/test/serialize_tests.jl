@testitem "Usage" begin
    using TokenizerStats: tracked_stats, serialize_usage!, usage_stats!
    using JLD2: jldopen
    using OnlineStats: Series

    # Create some stats
    ts = tracked_stats()
    for i in 1:10000
        code = rand(UInt32, rand(40:100))
        is_oov = rand(Bool)
        usage_stats!(ts, code, is_oov)
    end

    # Populate stats
    s = Series(;
        fertility=ts.fertility,
        nunique=ts.nunique,
        out_of_vocab=ts.out_of_vocab,
        ngrams=Series(ts.ngrams),
    )
    @test keys(s.stats) == keys(ts)
    @test length(s[:ngrams].stats) == length(ts[:ngrams])

    mktemp() do file, io
        # Precompile
        jldopen(file, "w") do f
            serialize_usage!(f, "val", s)
        end
        # Record serialization overhead
        stats = @timed jldopen(file, "w") do f
            serialize_usage!(f, "val", s)
        end

        # Check for regressions in serialization overhead
        io_bytes = stats.bytes
        file_bytes = stat(file).size
        mem_overhead = io_bytes / file_bytes
        @test mem_overhead < 4.5

        # Validate data
        jldopen(file, "r") do f
            @test Set(keys(f["val"])) == Set(["samples", string.(keys(s.stats))...])
            @test f["val"]["samples"] isa Int
            @test f["val"]["out_of_vocab"] isa Int
            @test f["val"]["nunique"] isa Dict{Int,Int}
            @test f["val"]["fertility"] isa Dict{Int,Int}
            @test length(f["val"]["ngrams"]) == length(s[:ngrams].stats)
        end
    end
end
