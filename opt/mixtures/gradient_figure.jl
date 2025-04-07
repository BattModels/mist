using Makie
using DataFrames
using CairoMakie: CairoMakie
using CSV: CSV
using CategoricalArrays: categorical, levelcode

include("../style.jl")
using .MISTStyle: MISTStyle, savefig, inch, pt

function figure_quiver()
    df_benzene = DataFrame(CSV.File("transferance_LiPF6_5.csv"))
    return df_benzene
end
