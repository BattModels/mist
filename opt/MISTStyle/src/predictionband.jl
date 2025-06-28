"""
    predictionband(center, lower, upper)

Plots a prediction band with median line overlayed
"""
@recipe(PredictionBand, x, center, lower, upper) do scene
    Theme(
        color=Makie.inherit(scene, (:Lines, :linecolor), :black),
        linewidth=Makie.inherit(scene, (:Lines, :linewidth), 2),
        band_color=(:blue, 0.1),
    )
end
function Makie.plot!(plt::PredictionBand)
    band!(plt, plt.x, plt.lower, plt.upper; color=plt.band_color)
    lines!(plt, plt.x, plt.center; color=plt.color, linewidth=plt.linewidth)
    return plt
end
