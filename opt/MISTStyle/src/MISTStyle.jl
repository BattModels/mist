module MISTStyle

using Makie
using CategoricalArrays: levels
using CairoMakie: CairoMakie
using StatsBase: StatsBase, AbstractWeights

# Conversion from units into pixels
const pt = 1
const px = (3/4) * pt
const inch = 72*pt
const mm = (1/25.4) * inch

export pt, inch, mm, sublabel!

""" Save duplicate figures for publication and web """
function savefig(name::String, f::Figure; dpi=300, fig_dir="fig")
    mkpath(dirname(joinpath(fig_dir, name)))
    save(joinpath(fig_dir, name * ".png"), f; px_per_unit=dpi / inch, backend=CairoMakie)
    save(joinpath(fig_dir, name * ".pdf"), f; pt_per_unit=pt, backend=CairoMakie)
    return nothing
end

function savefig(name::String; kwargs...)
    function curry_savefig(fig; kwargs...)
        savefig(name, fig; kwargs...)
        return fig
    end
    return curry_savefig
end

""" Helper function to get the label of an plot element """
label(x) = x.label[]

categorical_ticks(x) = (1:length(levels(x)), levels(x))


""" Get the colorbar attributes of a plot element """
function cb_attrs(cb::Colorbar)
    return (;
        colormap=cb.colormap,
        colorscale=cb.scale,
        colorrange=cb.colorrange,
        lowclip=cb.lowclip,
        highclip=cb.highclip,
    )
end
function cb_attrs(cb::Colorbar, plt)
    attrs = cb_attrs(cb)
    valid = Makie.attribute_names(plt)
    invalid = setdiff(keys(attrs), valid)
    return Base.structdiff(attrs, NamedTuple{(invalid...,)})
end

function parity_limits(x::AbstractVector, y::AbstractVector; inflate=0.05)
    l, u = extrema(Iterators.flatten((x, y)))
    limits = (l - inflate * (u - l), u + inflate * (u - l))
    return (limits, limits)
end


function sublabel!(f, letter; left=0, kwargs...)
    label_kwargs = (;
        fontsize=7pt,
        font=:bold,
        halign=:right,
        tellheight=false,
        padding=(0, left, 0, 0),
    )
    label_kwargs = merge(label_kwargs, kwargs)
    Label(f, "$letter)"; label_kwargs...)
end

include("errorcross.jl")
include("powerlaw.jl")
include("tantext.jl")
include("quadrant.jl")
include("asinh.jl")
include("sci_notation.jl")
include("nbins.jl")

const CAT_COLORS = cgrad(
    map(x -> RGBf(x ./ 255...), [
        (99, 110, 250),
        (239, 85, 59),
        (0, 204, 150),
        (171, 99, 250),
        (255, 161, 90),
        (25, 211, 243),
        (255, 102, 146),
        (182, 232, 128),
        (255, 151, 255),
        (254, 203, 82)
    ]),
    10
)

UM_COLORS = (;
    blue=colorant"#00274C",
    maize=colorant"#FFCB05",
    red=colorant"#9A3324",
    orange=colorant"#D86018",
    arboretum=colorant"#2F65A7",
    ash=colorant"#989C97",
    black=colorant"#131516",
)


const CONTINUOUS_COLORS = :lipari

function theme()
    Theme(
        rowgap=3pt,
        colgap=3pt,
        fonts=(;
            regular="Times New Roman Regular",
            bold="Times New Roman Bold",
        ),
        fontsize=8pt,
        size=(246, 152),
        figure_padding=(2, 2, 2, 2),
        colormap=:lipari,
        linewidth=1pt,
        CairoMakie=(;
            pt_per_unit=2,
            px_per_unit=300 / inch
        ),
        GLMakie=(; px_per_unit=4, scalefactor=4, focus_on_show=false),
        palette=(;
            color=CAT_COLORS,
            linestyle=[:solid, :dot, :dashdot],
        ),
        Lines=(;
            cycle=Cycle([:color, :linestyle], covary=true),
        ),
        GridLayout=(;
            default_rowgap=3pt,
            default_colgap=3pt,
        ),
        Axis=(;
            spinewidth=0.5,
            xlabelsize=6pt,
            ylabelsize=6pt,
            yticklabelsize=5pt,
            xticklabelsize=5pt,
            ylabelpadding=1pt,
            xlabelpadding=1pt,
            yticklabelpad=2pt,
            xticklabelpad=2pt,
            yticksize=2pt,
            ytickwidth=0.5pt,
            yminortickwidth=0.25pt,
            yminorticksize=1pt,
            xtickwidth=0.5pt,
            xticksize=2pt,
            xminortickwidth=0.25pt,
            xminorticksize=1pt,
            xgridwidth=0.5,
            ygridwidth=0.5,
            xminorgridwidth=0.5,
            yminorgridwidth=0.5,
            titlegap=2pt,
        ),
        Legend=(;
            titlegap=0,
            labelsize=5pt,
            patchsize=(6pt, 6pt),
            patchlabelgap=2pt,
            rowgap=0.5pt,
            colgap=1pt,
            groupgap=4pt,
            famevisible=true,
            framewidth=0.5,
            tellheight=false,
            tellwidth=false,
            padding=(1pt, 1pt, 1pt, 1pt),
            margin=(1pt, 1pt, 1pt, 1pt),
        ),
        Colorbar=(;
            spinewidth=0.5,
            tickwidth=0.5pt,
            ticksize=2pt,
            minorticksize=1pt,
            minortickwidth=0.25pt,
            labelsize=6pt,
            ticklabelsize=5pt,
            labelpadding=1pt,
            ticklabelpad=0pt,
            size=6pt,
        ),
        Scatter=(;
            markersize=5pt,
            marker=:x,
        ),
        BoxPlot=(;
            markersize=4pt,
        ),
        ErrorLines=(;
            whiskerwidth=3,
        ),
        ErrorCross=(;
            markersize=5pt,
            whiskerwidth=3,
        ),
        Quadrant=(;
            color=UM_COLORS.maize,
            linestyle=:solid,
            linecolor=UM_COLORS.blue,
            alpha=0.2,
        )
    )
end
end