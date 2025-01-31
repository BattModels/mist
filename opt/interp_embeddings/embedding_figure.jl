using Makie
using DataFrames
using CairoMakie: CairoMakie
using CSV: CSV
using CategoricalArrays: categorical, levelcode

include("../style.jl")
using .MISTStyle: MISTStyle, savefig, inch, pt

labels(x) = x.label[]

function figure_embedding()
    df_benzene = DataFrame(CSV.File("interp_benzene.csv"))
    df_condense = DataFrame(CSV.File("interp_condensed.csv"))

    f = Figure(; size=(3inch, 2inch))
    gl = GridLayout(f[1, 1])
    ax_aromatic = Axis(gl[1, 1])
    ax_rings = Axis(gl[2, 1]; limits=(nothing, (-2, nothing)))
    ax_condese = Axis(f[1, 2]; limits=(nothing, (1.0, nothing)))
    hidedecorations!(ax_aromatic)
    hidedecorations!(ax_rings)
    hidedecorations!(ax_condese)
    linkaxes!(ax_aromatic, ax_rings)
    rowgap!(gl, 2)

    aromaticity = categorical(df_benzene[!, "Aromaticity"])
    h = scatter!(ax_aromatic, df_benzene[!, "0"], df_benzene[!, "1"];
        color=levelcode.(aromaticity),
        colormap=MISTStyle.CAT_COLORS,
        colorrange=(1, 10),
    )
    elements = map(enumerate(levels(aromaticity))) do (i, label)
        MarkerElement(
            markersize=4pt,
            marker=h.marker,
            color=MISTStyle.CAT_COLORS[i],
            label=label
        )
    end
    Legend(gl[1, 1], elements, labels.(elements);
        labelsize=4pt,
        tellheight=false,
        tellwidth=false,
        padding=(1, 1, 1, 1),
        margin=(1, 1, 1, 1),
        patchlabelgap=0,
        rowgap=0,
        colgap=0,
        orientation=:horizontal,
        halign=:right,
        valign=:bottom,
        alignmode=Outside(),
    )



    h = scatter!(ax_rings, df_benzene[!, "0"], df_benzene[!, "1"];
        color=df_benzene[!, "Number of Benzene"],
        colormap=MISTStyle.CONTINUOUS_COLORS,
    )
    Colorbar(gl[3, 1], h;
        label="Number of Rings",
        halign=:right,
        valign=:bottom,
        tellheight=true,
        tellwidth=false,
        vertical=false,
        flipaxis=false,
        height=5pt,
    )
    condense = categorical(df_condense[!, "Condensed"])
    scatter!(ax_condese, df_condense[!, "0"], df_condense[!, "1"];
        color=levelcode.(condense),
        colormap=MISTStyle.CAT_COLORS,
        colorrange=(1, 10),
    )
    elements = map(enumerate(levels(condense))) do (i, label)
        MarkerElement(
            markersize=4pt,
            marker=h.marker,
            color=MISTStyle.CAT_COLORS[i],
            label=label
        )
    end
    Legend(f[1, 2], elements, labels.(elements);
        labelsize=4pt,
        tellheight=false,
        tellwidth=false,
        padding=(1, 1, 1, 1),
        margin=(1, 1, 1, 1),
        patchlabelgap=0,
        rowgap=0,
        colgap=0,
        # orientation=:horizontal,
        halign=:right,
        valign=:bottom,
        alignmode=Outside(),
    )


    label_kwargs = (;
        fontsize=6pt,
        font=:bold,
        halign=:right,
        tellheight=false,
    )
    Label(gl[1, 1, TopLeft()], "a)"; padding=(0, 3, 0, 5), label_kwargs...)
    Label(gl[2, 1, TopLeft()], "b)"; padding=(0, 3, 0, 5), label_kwargs...)
    Label(f[1, 2, TopLeft()], "c)"; padding=(0, 3, 0, 5), label_kwargs...)

    return f
end
