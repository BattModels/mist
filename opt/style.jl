module MISTStyle

using Makie

const pt = 4 / 3
const inch = 96

""" Save duplicate figures for publication and web """
function savefig(name::String, f::Figure; dpi=300, fig_dir="fig")
    mkpath(fig_dir)
    save(joinpath(fig_dir, name * ".pdf"), f; pt_per_unit=1)
    save(joinpath(fig_dir, name * ".png"), f; px_per_unit=dpi / inch)
    return nothing
end

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
            ylabelpadding=1pt,
            xlabelpadding=1pt,
            yticksize=3,
            ytickwidth=0.5,
            yminortickwidth=0.5,
            yminorticksize=2,
            xtickwidth=0.5,
            xticksize=3,
            xminortickwidth=0.5,
            xminorticksize=2,
            xgridwidth=0.5,
            ygridwidth=0.5,
            xminorgridwidth=0.5,
            yminorgridwidth=0.5,
            yticklabelsize=7pt,
            xticklabelsize=7pt,
        ),
        Legend=(;
            titlegap=0,
            patchsize=(8, 8),
            rowgap=2pt,
            colgap=8,
            groupgap=4pt,
            famevisible=true,
            framewidth=0.5,
            tellheight=false,
            tellwidth=false,
            padding=(2pt, 2pt, 2pt, 2pt),
        ),
        Colorbar=(;
            spinewidth=0.5,
            tickwidth=0.5,
            ticksize=2,
            labelsize=6pt,
            ticklabelsize=5pt,
            labelpadding=0pt,
            ticklabelpad=0pt,
        ),
        Scatter=(;
            markersize=5pt,
            marker=:x,
        ),
        ErrrorBar=(;
            whiskerwidth=2,
            linewidth=0.5,
        )
    )
end
end
