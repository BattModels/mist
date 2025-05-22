@testitem "help" begin
    using TokenizerStats: main
    Base.redirect_stdout(Pipe()) do
        @test main(["--help"]) == 0
    end
end

@testitem "parse_splits" begin
    using TokenizerStats: parse_splits
    @test Set(parse_splits("val")) == Set(["val"])
    @test Set(parse_splits("val,train")) == Set(["val", "train"])
    @test Set(parse_splits("all")) == Set(["train", "val", "test"])
end

