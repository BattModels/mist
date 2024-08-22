@testitem "log_smoothed_counts" begin
    using TokenizerStats: log_smoothed_counts
    @test log_smoothed_counts(1, 5, 256000) ≈ log(1 + BigInt(256000)^5)
    @test log_smoothed_counts(5_000_000, 1, 256000) ≈ log(5_000_000 + BigInt(256000)^1)
    @test log_smoothed_counts(0, 8, 256000) ≈ log(BigInt(256000)^8)
end
