@recipe(ErrorCross, x, y, error_x, error_y) do scene
    Attributes()
end

function Makie.plot!(plt::ErrorCross{<:NTuple{4,AbstractVector}})
    scatter!(plt, plt.x, plt.y; Makie.shared_attributes(plt, Scatter)...)
    h = errorbars!(plt, plt.x, plt.y, plt.error_y;
        direction=:y,
        Makie.shared_attributes(plt, Errorbars)...
    )
    errorbars!(plt, plt.x, plt.y, plt.error_x;
        color=h.color,
        direction=:x,
        Makie.shared_attributes(plt, Errorbars)...
    )
    return plt
end
