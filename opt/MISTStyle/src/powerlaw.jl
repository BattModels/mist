Makie.@recipe(Powerlaw, a, b) do scene
    Attributes(;
        npoints=100,
    )
end

xint(rect::Makie.Rect) = minimum(rect)[1] .. maximum(rect)[1]

function Makie.plot!(plt::Powerlaw)
    # Get Limits
    scene = Makie.parent_scene(plt)
    # limits = lift(xint, scene.finallimits)
    limits = lift(xint, Makie.projview_to_2d_limits(plt))

    # Regenerate points when the view / model updates
    points = Observable(Point2f[])
    onany(limits, plt[:a], plt[:b], plt[:npoints]) do limits, a, b, npoints
        # Sample x over the full plot width, plus a bit extra to avoid
        # clipping artifacts at the plot limits
        xmin = first(minimum(limits))
        xmax = first(maximum(limits))
        chrome = 2 * (xmax - xmin) / (npoints)
        transforms = (Makie.transform_func)(scene)
        xinv = Makie.inverse_transform(first(transforms))
        x = xinv.(range(xmin - chrome, xmax + chrome; length=npoints))
        y = map(x -> a * x^b, x)

        # Update points
        empty!(points[])
        append!(points[], Point2.(x, y))
        notify(points)
    end

    # Plot response, and translate it forward
    lines!(plt, points; Makie.shared_attributes(plt, Lines)...)
    return plt
end
