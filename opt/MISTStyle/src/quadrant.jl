@recipe(Quadrant, x, y, quad) do scene
    Attributes(;
        color=Makie.inherit(scene, (:Quadrant, :color), :blue),
        linecolor=Makie.inherit(scene, (:Quadrant, :linecolor), :black),
        linestyle=Makie.inherit(scene, (:Quadrant, :linestyle), :solid),
        linealpha=Makie.inherit(scene, (:Quadrant, :linealpha), 1),
        linewidth=Makie.inherit(scene, (:Quadrant, :linewidth), Makie.inherit(scene, :linewidth, 1)),
    )
end

function Makie.plot!(plt::Quadrant)
    ax= Makie.current_axis()
    limits = ax.finallimits
    p = lift(quadpoints, limits, plt[:x], plt[:y], plt[:quad])
    poly!(plt, p; color=plt[:color], Makie.shared_attributes(plt, Poly)...)

    attrs = Makie.shared_attributes(plt, HLines)
    attrs[:color] = plt[:linecolor]
    attrs[:alpha] = plt[:linealpha]
    hlines!(plt, plt[:y]; attrs...)

    attrs = Makie.shared_attributes(plt, VLines)
    attrs[:color] = plt[:linecolor]
    attrs[:alpha] = plt[:linealpha]
    vlines!(plt, plt[:x]; linewidth=5pt, attrs...)

    return plt
end

quadpoints(hr, x, y, quad) = quadpoints(corner(quad, hr), (x, y))
function quadpoints(p1::NTuple{2, <:Number}, p2::NTuple{2, <:Number})
    ll_x, ur_x = extrema(first, [p1, p2])
    ll_y, ur_y = extrema(last, [p1, p2])
    return [(ll_x, ll_y), (ur_x, ll_y), (ur_x, ur_y), (ll_x, ur_y)]
end
function corner(quad::Symbol, hr)
    if quad == :lt
        return tuple(hr.origin .+ (0, hr.widths[2])...)
    elseif quad == :rt
        return tuple(hr.origin .+ hr.widths...)
    elseif quad == :lb
        return tuple(hr.origin...)
    elseif quad == :rb
        return tuple(hr.origin .+ (hr.widths[1], 0)...)
    else
        error("Quadrant $quad not recognized")
    end
end

