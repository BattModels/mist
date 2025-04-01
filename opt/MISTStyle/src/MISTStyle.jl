module MISTStyle

using Makie
using CategoricalArrays: levels
using GLMakie: GLMakie
using CairoMakie: CairoMakie

const pt = 3 / 4
const inch = 96

export pt, inch

""" Save duplicate figures for publication and web """
function savefig(name::String, f::Figure; dpi=300, fig_dir="fig")
    mkpath(fig_dir)
    save(joinpath(fig_dir, name * ".pdf"), f; pt_per_unit=1, backend=CairoMakie)
    save(joinpath(fig_dir, name * ".png"), f; px_per_unit=dpi / inch, backend=GLMakie)
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


function sublabel!(f, letter; left=0, kwargs...)
    label_kwargs = (;
        fontsize=8pt,
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
    blue=RGBf(0, 39 / 255, 76 / 255),
    maize=RGBf(1, 203 / 255, 5 / 255),
)


const CONTINUOUS_COLORS = :lipari

function theme()
    Theme(
        rowgap=2,
        colgap=2,
        fonts=(;
            regular="Times New Roman Regular",
            bold="Times New Roman Bold",
        ),
        fontsize=8pt,
        size=(246, 152),
        figure_padding=(2, 2, 2, 2),
        colormap=:lipari,
        linewidth=0.5,
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
        Axis=(;
            spinewidth=0.5,
            xlabelsize=8pt,
            ylabelsize=8pt,
            yticklabelsize=6pt,
            xticklabelsize=6pt,
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
        ),
        Legend=(;
            titlegap=0,
            labelsize=8pt,
            patchsize=(8pt, 8pt),
            patchlabelgap=3pt,
            rowgap=1pt,
            colgap=3pt,
            groupgap=4pt,
            famevisible=true,
            framewidth=0.5,
            tellheight=false,
            tellwidth=false,
            padding=(2pt, 2pt, 2pt, 2pt),
            margin=(2pt, 2pt, 2pt, 2pt),
        ),
        Colorbar=(;
            spinewidth=0.5,
            tickwidth=0.5,
            ticksize=2,
            labelsize=8pt,
            ticklabelsize=6pt,
            labelpadding=0pt,
            ticklabelpad=0pt,
            size=8pt,
        ),
        Scatter=(;
            markersize=5pt,
            marker=:x,
        ),
        ErrorLines=(;
            whiskerwidth=3,
        ),
        ErrorCross=(;
            markersize=5pt,
            whiskerwidth=3,
        )
    )
end
end
