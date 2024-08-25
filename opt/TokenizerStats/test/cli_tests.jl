@testitem "help" begin
    using TokenizerStats: main
    Base.redirect_stdout(Pipe()) do
        @test main(["--help"]) == 0
    end
end
