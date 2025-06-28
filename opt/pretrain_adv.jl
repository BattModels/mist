using Makie
using MISTStyle
using JSON
using DataFrames
using CategoricalArrays

Makie.set_theme!(MISTStyle.theme())

df = DataFrame(JSON.parsefile(joinpath(@__DIR__, "pretrain_adv.json")))
df.model_size = categorical(df.model_size; levels=["28M", "1.8B"])

f = Figure(; size=(275, 165))
ax = Axis(f[1, 1];
    xlabel="Dataset Size",
    ylabel="Validation Avg. MAE Loss",
)

linewidth = 2pt
markersize = 1.5pt
for row in eachrow(df)
    scatterlines!(ax, row.dataset, row.loss;
        color=MISTStyle.CAT_COLORS[levelcode(row.model_size)],
        marker=row.pretrained ? :circle : :rect,
        linewidth,
    )
end

contentgroups = [
    map([("Pretrained", :circle), ("Random", :rect)]) do (label, marker)
        MarkerElement(; label, marker, color=:black)
    end,
    map(enumerate(unique(df.model_size))) do (idx, label)
        LineElement(; label, color=MISTStyle.CAT_COLORS[idx], linewidth)
    end,
]

Legend(f[1,1], contentgroups, MISTStyle.label(contentgroups), ["Model Initialization", "Encoder Size"];
    nbanks=2,
    halign=:right,
    valign=:top,
)

MISTStyle.savefig("pretrain_adv", f)
f
