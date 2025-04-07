Makie.@recipe(TanText, h, x) do scene
    Attributes(;
        text = "",
        fontsize = Makie.theme(scene, :fontsize),
        delta = 1,
    )
end

function Makie.plot!(plt::TanText)
    scene = Makie.get_scene(plt)
    m = lift(get_grade, plt[:h])
    x = plt[:x]
    y = lift(get_position, plt[:h], plt[:x])
    Δx = plt[:delta]
    rotation = lift(scene.camera.projection, scene.viewport, x, y, m, Δx) do p, vp, x, y, m, Δx
        p1 = Makie.project(scene, :data, :pixel, (x, y))
        p2 = Makie.project(scene, :data, :pixel, (x + Δx, y + m*Δx))
        atan(p2[2] - p1[2], p2[1] - p1[1])
    end
    @show rotation

    attrs = Makie.shared_attributes(plt, Makie.Text)
    Makie.text!(plt, x, y; rotation, fontsize=plt[:fontsize], attrs...)
end

get_grade(h::Makie.ABLines) = lift(first, h[2])[]
get_grade(h) = error("$(typeof(h)) is not supported")


function get_position(h::Makie.ABLines, x::Number)
    b = lift(first, h[1])[]
    m = lift(first, h[2])[]
    return m*x + b
end
