@recipe(ErrorCross, x, y, error_x, error_y) do scene
    Attributes(;
        n_stds=3,
        markervisible=true,
    )
end

function Makie.plot!(plt::ErrorCross{<:NTuple{4,AbstractVector}})
    scatter!(plt, plt.x, plt.y;
        visible=plt.markervisible,
        Makie.shared_attributes(plt, Scatter)...
    )
    error_x = @lift $(plt.error_x) * first($(plt.n_stds))
    error_y = @lift $(plt.error_y) * last($(plt.n_stds))
    h = errorbars!(plt, plt.x, plt.y, error_y;
        direction=:y,
        Makie.shared_attributes(plt, Errorbars)...
    )
    errorbars!(plt, plt.x, plt.y, error_x;
        color=h.color,
        direction=:x,
        Makie.shared_attributes(plt, Errorbars)...
    )
    return plt
end
