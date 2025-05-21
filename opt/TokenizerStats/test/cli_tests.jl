@testitem "help" begin
    using TokenizerStats: main
    Base.redirect_stdout(Pipe()) do
        @test main(["--help"]) == 0
    end
end

@testitem "parse_splits" begin
    using TokenizerStats: parse_splits
    @test parse_splits("val") == ["val"]
    @test parse_splits("val,train") == ["val", "train"]
    @test parse_splits("all") == ["train", "val", "test"]
end

