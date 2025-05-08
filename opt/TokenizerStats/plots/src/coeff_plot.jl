struct CoeffInternal{T}
    expectation::T
    credible_interval::NTuple{2,T}
end

lower(x::CoeffInternal) = first(x.credible_interval)
upper(x::CoeffInternal) = last(x.credible_interval)
StatsBase.mean(x::CoeffInternal) = x.expectation

function Base.:-(x::CoeffInternal, y::CoeffInternal)
    μ = mean(x) - mean(y)
    lb = lower(x) - upper(y)
    ub = upper(x) - lower(y)
    ci = lb <= ub ? (lb, ub) : (ub, lb)
    return CoeffInternal(μ, ci)
end
Base.:-(::CoeffInternal, ::Missing) = missing
Base.:-(::Missing, ::CoeffInternal) = missing
Base.:/(::Missing, ::CoeffInternal) = missing
Base.:*(x::CoeffInternal, y::Real) = CoeffInternal(y * mean(x), y .* x.credible_interval)
Base.:*(x::Real, y::CoeffInternal) = y * x

function Base.:/(x::CoeffInternal, y::CoeffInternal)
    μ = mean(x) / mean(y)
    limits = map(x -> /(x...), Iterators.product(x.credible_interval, y.credible_interval))
    return CoeffInternal(μ, extrema(limits))
end

function coefint(model::M; level=0.95) where {M<:StatsBase.StatisticalModel}
    μ = coef(model)
    ci = map(x -> tuple(x...), eachrow(confint(model; level)))
    return CoeffInternal.(μ, ci)
end

function coefint(model, names::Vector{String}; kwargs...)
    ci = Dict(zip(coefnames(model), coefint(model)))
    return [get(ci, name, missing) for name in names]
end

@recipe(EffectBars, val, effect)  do scene
    Theme(
        color = Makie.inherit(scene, :linecolor, :black),
        colormap = Makie.inherit(scene, :colormap, :tab10),
        direction = :x,
        whiskerwidth = Makie.inherit(scene, (:RangeBars, :whiskerwidth), 10),
        gap = 0.2,
        dodge = Makie.automatic,
        n_dodge = Makie.automatic,
        dodge_gap = 0.03,
        noeffect_visible = true,
        noeffect_color = :black,
        noeffect_linewidth = Makie.inherit(scene, :linewidth, 1),
        noeffect_linestyle = :dash,
    )
end

function Makie.plot!(plt::EffectBars)
    effect = plt.effect
    highs = @lift(map(upper, $effect))
    lows = @lift(map(lower, $effect))

    val = first(Makie.compute_x_and_width(plt.val[], 1, plt.gap[], plt.dodge[], plt.n_dodge[], plt.dodge_gap[]))

    if plt[:noeffect_visible][]
        vlines!(plt, 0;
            color=plt[:noeffect_color],
            linewidth=plt[:noeffect_linewidth],
            linestyle=plt[:noeffect_linestyle],
        )
    end

    rangebars!(plt, val, lows, highs;
        color = plt[:color],
        colormap=plt[:colormap],
        direction = plt[:direction],
        whiskerwidth = plt[:whiskerwidth],
        Makie.shared_attributes(plt, Makie.Rangebars)...
    )
    expecation = lift(plt[:direction], effect, val) do dir, effect, val
        if dir == :x
            return Point2.(mean.(effect), val)
        else
            return Point2.(val, mean.(effect))
        end
    end
    scatter!(plt, expecation;
        color=plt[:color],
        colormap=plt[:colormap],
        Makie.shared_attributes(plt, Scatter)...
    )



    return plt
end


effect_size_plot(models; kwargs...) = effect_size_plot!(Figure(), models; kwargs...)
function effect_size_plot!(f::Figure, models; relative=true)
    @show names = intersect(coefnames.(models)...)
    ci = map(models) do model
        cdx = map(in(names), coefnames(model))
        return coefint(model)[cdx]
    end
    ci = hcat(ci...)
    if relative
        ci = (ci .- ci[[1], :]) ./ ci[[1], :]
        ci = ci[2:end, :]
        names = names[2:end]
    end

    ax = Axis(f[1, 1];
        yticks=(eachindex(names), names),
    )
    ci = vec(ci)
    val = vec(repeat(1:length(names), 1, length(models)))
    dodge = vec(repeat((1:length(models))', length(names)))

    effectbars!(ax, val, vec(ci); dodge, color=dodge)

    return f
end
