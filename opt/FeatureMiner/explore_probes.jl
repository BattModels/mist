using Makie
using DataFrames
using CSV
using FeatureMiner: load_fitted_probes


df = DataFrame()
for ckpt in readdir(joinpath(@__DIR__, "..", "..", "linear-probes"); join=true)
    isdir(ckpt) || continue
    isdir(joinpath(ckpt, "checkpoints")) || continue

    ckpt_probes = load_fitted_probes(ckpt)
    select!(ckpt_probes, Not(:weight))
    select!(ckpt_probes, Not(:Bias))
    append!(df, ckpt_probes)
end
CSV.write("linear_probes.csv", df)
