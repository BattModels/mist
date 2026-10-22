@recipe ErrorLines (x, y, error_y) begin
    n_stds = 3
    color = @inherit linecolor
    linewidth = @inherit linewidth
    colormap = @inherit colormap
    colorrange = Makie.automatic
end
Makie.convert_arguments(::Type{<:ErrorLines}, x::Any, y::AbstractVector{<:UQReal}) = (x, mean.(y), std.(y))
Makie.convert_arguments(::Type{<:ErrorLines}, x::Any, y::AbstractVector{<:Real}) = (x, y, zero(y))

function Makie.plot!(plt::ErrorLines{<:Tuple{AbstractVector,AbstractVector{<:Real},AbstractVector{<:Real}}})
    # Forward the parent's inputs explicitly: Makie's computed defaults in
    # shared_attributes otherwise replace group colors with the line default.
    lines!(plt, plt.x, plt.y;
        Makie.shared_attributes(plt, Lines)...,
        color=plt.color, colormap=plt.colormap, colorrange=plt.colorrange,
    )
    if !isnothing(plt.error_y)
        error = @lift $(plt[:error_y]) * $(plt[:n_stds])
        errorbars!(plt, plt.x, plt.y, error;
            Makie.shared_attributes(plt, Makie.Errorbars)...,
            color=plt.color, colormap=plt.colormap, colorrange=plt.colorrange,
        )
    end
    return plt
end

Makie.convert_arguments(::Type{<:ErrorCross}, x::AbstractVector{<:UQReal}, y::AbstractVector{<:UQReal}) = (mean.(x), mean.(y), stderror.(x), stderror.(y))
