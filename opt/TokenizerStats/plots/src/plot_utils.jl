function theme()
    Theme(
        rowgap=2,
        colgap=2,
        fonts=(;
            :regular => "Times New Roman Regular",
            :bold => "Times New Roman Bold",
            :italic => "Times New Roman Italic",
        ),
        fontsize=8pt,
        size=(246, 152),
        figure_padding=(1, 15, 2, 2),
        CairoMakie=(;
            pt_per_unit=2,
            px_per_unit=300 / inch
        ),
        GLMakie=(; px_per_unit=4, scalefactor=4, focus_on_show=false),
        palette=(;
            color=cgrad(:seaborn_muted, 10),
            linestyle=[:solid, :dot, :dashdot],
        ),
        Lines=(;
            cycle=Cycle([:color, :linestyle], covary=true),
        ),
        markersize=4pt,
        linewidth=1pt,
        Axis=(;
            titlegap=2pt,
            spinewidth=0.5,
            ylabelpadding=3pt,
            yticksize=3,
            ytickwidth=0.5,
            yminortickwidth=0.5,
            yminorticksize=2,
            xtickwidth=0.5,
            xticksize=3,
            xminortickwidth=0.5,
            xminorticksize=1.5,
            xgridwidth=0.5,
            ygridwidth=0.5,
            xminorgridwidth=0.5,
            yminorgridwidth=0.5,
        ),
        Legend=(;
            titlegap=0,
            patchsize=(8, 8),
            rowgap=2pt,
            colgap=8,
            groupgap=4pt,
            framewidth=0.5,
            tellheight=false,
            tellwidth=false,
            padding=(2pt, 2pt, 2pt, 2pt),
        ),
        Colorbar=(;
            spinewidth=0.5,
            tickwidth=0.5,
            ticksize=2,
        ),
        Scatter=(;
            markersize=8pt,
            marker=:x,
        ),
        ErrrorBar=(;
            whiskerwidth=2,
            linewidth=0.5,
        ),
        EffectBars=(;
            linewidth=1pt,
            marker=:circle,
            markersize=3pt,
            colormap=[:red, :blue],
            whiskerwidth=8pt,
            noeffect_linewidth=1pt,
        ),
    )
end

struct Asinh
    a::Float64
end
Asinh() = Asinh(1)
(m::Asinh)(x::Real) = m.a * asinh(x / m.a)

Makie.inverse_transform(m::Asinh) = x -> m.a * sinh(x / m.a)
Makie.defined_interval(::Asinh) = Makie.defined_interval(identity)
Makie.defaultlimits(m::Asinh) = (0.0, 10 * m.a)

my_sqrt(x) = sqrt(x)
Makie.inverse_transform(::typeof(my_sqrt)) = x -> x^2
Makie.defaultlimits(::typeof(my_sqrt)) = (0.0, 10.0)
Makie.defined_interval(::typeof(my_sqrt)) = Makie.defined_interval(sqrt)

Makie.inverse_transform(::typeof(asinh)) = sinh
Makie.defined_interval(::typeof(asinh)) = Makie.defined_interval(identity)
Makie.defaultlimits(::typeof(asinh)) = (0.0, 10.0)

labels(x) = map(e -> e.label[], x)

# Estimate Number of Histogram Bins from data
hist_nbins(x::AbstractVector, w::AbstractWeights) = hist_nbins(:scott, x)
hist_nbins(method::Symbol, x::AbstractVector) = hist_nbins(method, x, StatsBase.UnitWeights{Int}(length(x)))

hist_nbins(method, x, w) = hist_nbins(Val(Symbol(method)), x, w)
hist_nbins(method::Symbol, x::AbstractVector, w::AbstractWeights) = hist_nbins(Val(method), x, w)
function hist_nbins(::Val{:scott}, x, w)
    σ = std(x, w)
    h = 3.5 * σ / cbrt(length(x))
    n = (maximum(x) - minimum(x)) / h
    return ceil(Int, n)
end
hist_nbins(::Val{:sqrt}, x, w) = ceil(Int, sqrt(length(x)))
hist_nbins(::Val{M}, args...) where {M} = MethodError(hist_nbins, M, args...)
hist_nbins(::Val{:sturges}, x, w) = ceil(Int, log2(length(x)) + 1)
hist_nbins(::Val{:sturges}, x, w::StatsBase.FrequencyWeights) = ceil(Int, log2(sum(w)) + 1)

function tokenizer_label(tok; vocab_size=false)
    name = tok.name
    if match(r"smirk-gpe", name) !== nothing
        name = "smirk-gpe"
    end
    if vocab_size
        name *= "\n$(tok.vocab_size) tokens"
    end
    return name
end

@recipe(TokenUsage, usage, vocab_size) do scene
    Attributes(
        smoothing=1,
    )
end

function collate_token_usage(ids::AbstractVector{<:Integer}, counts::Dict{<:AbstractString,Any}; smoothing=1)
    counts = Dict((parse(Int, k) => v for (k, v) in pairs(counts)))
    return collate_token_usage(ids, counts; smoothing)
end
collate_token_usage(counts::Dict, vocab_size::Integer; kwargs...) = collate_token_usage(collect(0:vocab_size-1), counts; kwargs...)
function collate_token_usage(ids::AbstractVector{T}, counts::Dict; smoothing) where {T}
    usage = Vector{Int}(undef, length(ids))
    for (idx, token_id) in enumerate(ids)
        usage[idx] = get(counts, token_id, 0) + smoothing
    end
    return usage
end

function Makie.plot!(plt::TokenUsage)
    vocab_size = plt.vocab_size[]
    ids = 0:vocab_size-1
    usage = collate_token_usage(ids, plt.usage[]; smoothing=plt[:smoothing][])
    usage = usage ./ sum(usage)
    sort!(usage; rev=true)
    I = -log.(usage)
    x = range(0, 1; length=length(I)) |> reverse
    attr = Makie.shared_attributes(plt, Stairs)
    stairs!(plt, I; attr...)
end

Makie.@recipe(DodgedErrorBars, x, y, error) do scene
    Attributes(
        width=Makie.inherit(scene, :BarPlot, :width),
        dodge=Makie.inherit(scene, :BarPlot, :dodge),
        n_dodge=Makie.inherit(scene, :BarPlot, :n_dodge),
        gap=Makie.inherit(scene, :BarPlot, :gap),
        dodge_gap=Makie.inherit(scene, :BarPlot, :dodge_gap),
        direction=:y,
    )
end

function Makie.plot!(plt::DodgedErrorBars)
    x = lift(plt[:x], plt[:width], plt[:gap], plt[:dodge], plt[:n_dodge], plt[:dodge_gap]) do x, width, gap, dodge, n_dodge, dodge_gap
        first(Makie.compute_x_and_width(x, width, gap, dodge, n_dodge, dodge_gap))
    end
    attr = Makie.shared_attributes(plt, Errorbars)
    if plt[:direction][] == :y
        errorbars!(plt, x, plt[:y], plt[:error]; direction=:y, attr...)
    else
        errorbars!(plt, plt[:y], plt[:x], plt[:error]; direction=:x, attr...)
    end
    return plt
end


function findfont(name, style)
    for folder in FreeTypeAbstraction.fontpaths()
        for file in readdir(folder; join=true)
            first(splitext(basename(file))) == name || continue
            n_faces = FTFont(newface(file, -1)).num_faces
            for face in range(0; length=n_faces)
                font = FTFont(newface(file, face))
                if font.style_name == style
                    return font
                end
            end
        end
    end
    return nothing
end

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

    # Generate points to plot
    notify(limits)

    # Plot response, and translate it forward
    lines!(plt, points; Makie.shared_attributes(plt, Lines)...)
    return plt
end

function siglevel(p::Real; cutoff=[0.05, 0.01, 0.001], symbol="*")
    l = findlast(sort(cutoff; rev=true) .>= p)
    return isnothing(l) ? "" : symbol ^ l
end

function label_tokenizers!(df)
    tokenizer_classes = OrderedDict(
        "nlp" => "NLP",
        "character" => "Character",
        "unigram" => "Unigram",
        "bpe" => "BPE",
        "atomwise" => "Atom-wise",
        "spe" => "SPE/APE",
        "smirk-gpe" => "Smirk-GPE",
        "smirk" => "Smirk",
    )
    df.tokenizer_class = map(df.tokenizer_domain, df.tokenizer_class) do domain, tokenizer_class
        if domain == "chemistry"
            return tokenizer_class
        elseif domain in ["nlp", "nlp-science"]
            return "nlp"
        else
            return "$tokenizer_class, $domain"
        end
    end
    ckeys = collect∘keys
    subset!(df, :tokenizer_class => ByRow(in(ckeys(tokenizer_classes))))
    transform!(df, :tokenizer_class => ByRow(x -> tokenizer_classes[x]) => :tokenizer_class)
    df.tokenizer_class = categorical(df.tokenizer_class, levels=(collect∘values)(tokenizer_classes), ordered=true)
    return df
end

function label_datasets!(df)
    df.dataset = map(ds -> ds in ["realspace", "tmQM"] ? ds : "MoleculeNet", df.dataset)
    df.dataset = map(ds -> ds == "realspace" ? "REALSpace" : ds, df.dataset)
    df.dataset = categorical(df.dataset, levels=["REALSpace", "tmQM", "MoleculeNet"], ordered=true)
    return df
end

categorical_ticks(x) = (1:length(levels(x)), levels(x))

function barploterrors!(ax, x, y, dodge; std=nothing, colormap, colorrange=nothing)
    if isnothing(colorrange)
        colorrange = extrema(levelcode.(dodge))
    end

    h = barplot!(ax, levelcode.(x), y;
        dodge=levelcode.(dodge),
        colormap,
        colorrange,
        color=levelcode.(dodge),
    )

    if !isnothing(std)
        SmirkPaperPlots.dodgederrorbars!(ax, levelcode.(x), y, std;
            dodge=h.dodge,
            width=h.width,
            n_dodge=h.n_dodge,
            gap=h.gap,
            dodge_gap=h.dodge_gap,
            linewidth=1,
            color=:black,
        )
    end

    return h
end


