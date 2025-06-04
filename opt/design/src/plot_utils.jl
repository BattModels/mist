@recipe(ErrorLines, x, y, error_y) do scene
    Attributes(;
        n_stds=3,
    )
end
Makie.convert_arguments(::Type{<:ErrorLines}, x::Any, y::AbstractVector{<:UQReal}) = (x, mean.(y), std.(y))
Makie.convert_arguments(::Type{<:ErrorLines}, x::Any, y::AbstractVector{<:Real}) = (x, y, zero(y))

function Makie.plot!(plt::ErrorLines{<:Tuple{AbstractVector,AbstractVector{<:Real},AbstractVector{<:Real}}})
    lines!(plt, plt.x, plt.y; Makie.shared_attributes(plt, Lines)...)
    if !isnothing(plt.error_y)
        error = @lift $(plt[:error_y]) * $(plt[:n_stds])
        errorbars!(plt, plt.x, plt.y, error;
            Makie.shared_attributes(plt, Makie.Errorbars)...
        )
    end
    return plt
end

Makie.convert_arguments(::Type{<:ErrorCross}, x::AbstractVector{<:UQReal}, y::AbstractVector{<:UQReal}) = (mean.(x), mean.(y), stderror.(x), stderror.(y))
