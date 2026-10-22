Makie.@recipe(TanText, h, pos) do scene
    Attributes(;
        text="",
        fontsize=Makie.theme(scene, :fontsize),
        delta=1,
    )
end

function Makie.plot!(plt::TanText)
    scene = Makie.get_scene(plt)

    # Keep the referenced plot inside Makie's compute graph. Converting it to
    # an Observable via lift would try to deepcopy its entire scene in 0.24.
    map!(get_position, plt, [:h, :pos], :text_origin)
    map!(get_rot2, plt, [:h, :pos], :text_tangent)
    origin = plt[:text_origin]
    h_rot = plt[:text_tangent]

    rotation = lift(origin, h_rot, scene.camera.projection, scene.viewport) do origin, h_rot, _, _
        p1 = Makie.project(scene, :data, :pixel, origin)
        p2 = Makie.project(scene, :data, :pixel, origin .+ h_rot)
        atan(p2[2] - p1[2], p2[1] - p1[1])
    end
    attrs = Makie.shared_attributes(plt, Makie.Text)
    Makie.text!(plt, origin;
        fontsize=plt[:fontsize],
        rotation,
        attrs...
    )
    return plt
end

get_rot2(h, ::Number) = get_rot2(h)
get_rot2(h::Makie.ABLines) = (1, only(h[2][]))
get_rot2(::Makie.HLines) = (1, 0)
get_rot2(::Makie.VLines) = (0, 1)
get_rot2(h) = error("$(typeof(h)) is not supported")

function get_position(h::Makie.ABLines, x::Number)
    m = only(h[2][])
    b = only(h[1][])
    Makie.Point2(x, m * x + b)
end

get_position(h::Makie.HLines, x::Number) = Makie.Point2(x, only(h[1][]))
get_position(h::Makie.VLines, x::Number) = Makie.Point2(only(h[1][]), x)
