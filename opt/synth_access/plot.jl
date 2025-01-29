using CairoMakie

include("../style.jl")
using .MISTStyle

function figure_synth_access(df)
    f = Figure()
    ax = Axis(f[1, 1]; ylabel="AUROC")

end
