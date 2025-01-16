@testitem "Usage" begin
    using TokenizerStats: tracked_stats, serialize_usage!
    using JLD2: jldopen
    using OnlineStats: Series

    # Create some stats 
    ts = tracked_stats()
    s = Series(;
        fertility=ts.fertility,
        nunique=ts.nunique,
        out_of_vocab=ts.out_of_vocab,
        ngrams=Series(ts.ngrams),
    )
    @test keys(s.stats) == keys(ts)
    @test length(s[:ngrams].stats) == length(ts[:ngrams])

    mktemp() do file, io
        jldopen(file, "w") do f
            serialize_usage!(f, "val", s)
        end
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
