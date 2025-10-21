using Makie
using DataFrames
using Random
using Statistics
using UMAP
using Distances
using CairoMakie: CairoMakie
using CSV: CSV
using CategoricalArrays: categorical, levelcode
using MISTStyle: MISTStyle, savefig, inch, pt, label

function figure_embedding()
    df_benzene = DataFrame(CSV.File("interp_benzene.csv"))
    df_condense = DataFrame(CSV.File("interp_condensed.csv"))
    figure_embedding(df_benzene, df_condense)
end

function figure_embedding(df_benzene, df_condense)

    f = Figure(; size=(3inch, 2inch))
    gl = GridLayout(f[1, 1])
    ax_aromatic = Axis(gl[1, 1])
    ax_rings = Axis(gl[2, 1]; limits=(nothing, (-2.3, nothing)))
    ax_condese = Axis(f[1, 2]; limits=(nothing, (0.8, nothing)))
    hidedecorations!(ax_aromatic)
    hidedecorations!(ax_rings)
    hidedecorations!(ax_condese)
    linkaxes!(ax_aromatic, ax_rings)
    rowgap!(gl, 2)

    aromaticity = categorical(df_benzene[!, "Aromaticity"])
    h = scatter!(ax_aromatic, df_benzene[!, "0"], df_benzene[!, "1"];
        color=levelcode.(aromaticity),
        colormap=MISTStyle.CAT_COLORS,
        colorrange=(1, 10),
        marker=:circle,
        markersize=3pt,
    )
    elements = map(enumerate(levels(aromaticity))) do (i, label)
        MarkerElement(
            markersize=4pt,
            marker=h.marker,
            color=MISTStyle.CAT_COLORS[i],
            label=label
        )
    end
    Legend(gl[1, 1], elements, label.(elements);
        labelsize=5pt,
        tellheight=false,
        tellwidth=false,
        padding=(1, 1, 1, 1),
        margin=(1, 1, 1, 1),
        patchlabelgap=0,
        rowgap=0,
        colgap=0,
        # orientation=:horizontal,
        halign=:right,
        valign=:bottom,
        alignmode=Outside(),
    )


    h = scatter!(ax_rings, df_benzene[!, "0"], df_benzene[!, "1"];
        color=df_benzene[!, "Number of Benzene"],
        colormap=MISTStyle.CONTINUOUS_COLORS,
        marker=:circle,
        markersize=3pt,
    )
    Colorbar(gl[3, 1], h;
        label="Number of Rings",
        halign=:right,
        valign=:bottom,
        tellheight=true,
        tellwidth=false,
        vertical=false,
        flipaxis=false,
        height=5pt,
    )
    condense = categorical(df_condense[!, "Condensed"])
    scatter!(ax_condese, df_condense[!, "0"], df_condense[!, "1"];
        color=levelcode.(condense),
        colormap=MISTStyle.CAT_COLORS,
        colorrange=(1, 10),
        marker=:circle,
        markersize=3pt,
    )
    elements = map(enumerate(levels(condense))) do (i, label)
        MarkerElement(
            markersize=h.markersize,
            marker=h.marker,
            color=MISTStyle.CAT_COLORS[i],
            label=label
        )
    end
    Legend(f[1, 2], elements, label.(elements);
        tellheight=false,
        tellwidth=false,
        padding=(1, 1, 1, 1),
        margin=(1, 1, 1, 1),
        patchlabelgap=0,
        rowgap=0,
        colgap=0,
        orientation=:horizontal,
        halign=:right,
        valign=:bottom,
        alignmode=Outside(),
    )


    label_kwargs = (;
        fontsize=6pt,
        font=:bold,
        halign=:right,
        tellheight=false,
    )
    Label(gl[1, 1, TopLeft()], "a)"; padding=(0, 3, 0, 5), label_kwargs...)
    Label(gl[2, 1, TopLeft()], "b)"; padding=(0, 3, 0, 5), label_kwargs...)
    Label(f[1, 2, TopLeft()], "c)"; padding=(0, 3, 0, 5), label_kwargs...)

    return f
end

function figure_olfactory()
    # Load condensed-phase coordinates
    df_condense = DataFrame(CSV.File("interp_olfactory.csv"))

    # --- Figure & single axis ------------------------------------------------
    f  = Figure(; size = (1.5inch, 1.5inch))
    ax = Axis(f[1, 1]; limits = (nothing,  (-25, 26)))
    hidedecorations!(ax)                # hides ticks, spines, labels

    # --- Scatter plot --------------------------------------------------------
    condense = categorical(df_condense[!, "label"])
    n = length(levels(condense))
    palette = MISTStyle.CAT_COLORS[1:n]     # exactly n colours

    h = scatter!(
        ax,
        df_condense[!, "0"], df_condense[!, "1"];
        color      = levelcode.(condense),
        colormap   = palette,
        colorrange = (1, n),
        marker     = :circle,
        markersize = 5pt,
    )

    # --- Legend --------------------------------------------------------------
    elements = map(enumerate(levels(condense))) do (i, label)
        MarkerElement(
            markersize = 4pt,
            marker     = h.marker,
            color      = MISTStyle.CAT_COLORS[i],
            label      = label,
        )
    end

    Legend(
        f[1, 1],
        elements,
        label.(elements);   # show category names
        labelsize      = 9pt,
        tellheight     = false,
        tellwidth      = false,
        padding        = (1, 1, 1, 1),
        margin         = (1, 1, 1, 1),
        patchlabelgap  = 0,
        patchsize = (9, 5),
        rowgap         = 0,
        colgap         = 0,
        orientation    = :horizontal,
        halign         = :right,
        valign         = :bottom,
        alignmode      = Outside(),
    )

    return f
end


function classify_decay_from_NZ(N::Real, Z::Real;
    tol_offset::Real=0.8, tol_scale::Real=0.02)

    # Beta-stability valley (from SEMF): Z_beta(A) ≈ A / (2 + 0.015 * A^(2/3))
    predict_Z_beta(A::Real) = A / (2 + 0.015 * A^(2/3))

    A = N + Z
    # very heavy nuclei
    if Z ≥ 92 && A ≥ 240
        return "Spontaneous \nFission"
    elseif Z ≥ 84 && A ≥ 210
        return L"$\alpha$"
    end
    zβ = predict_Z_beta(A)
    δ  = Z - zβ
    tol = tol_offset + tol_scale * Z  # widen tolerance for heavier Z
    if abs(δ) ≤ tol
        return "Stable"
    elseif δ < 0
        return L"$\beta^-$"   # too many neutrons -> beta- decay
    else
        return L"$\beta^+$"   # too many protons -> beta+/EC
    end
end

function figure_isotopes_umap(n_neighbors::Int=10, min_dist::Real=2, metric::Symbol=:manhattan, seed::Int=42)

    csv_path = "isotope_embeddings.csv"
    df = DataFrame(CSV.File(csv_path))

    # Labels from N/Z
    N = Float64.(coalesce.(df.neutrons, NaN))
    Z = Float64.(coalesce.(df.protons, NaN))
    labels_raw = [classify_decay_from_NZ(N[i], Z[i]) for i in eachindex(N)]

    # Embedding matrix: numeric columns excluding smiles/N/Z
    exclude = Set([:smiles, :neutrons, :protons])
    _is_numeric_col(col) = (eltype(col) <: Real) || (eltype(col) <: Union{Missing,Real})
    embed_cols = [nm for nm in names(df) if nm ∉ exclude && _is_numeric_col(df[!, nm])]
    isempty(embed_cols) && error("No numeric embedding columns found (after excluding smiles/N/Z).")

    X = Matrix{Float64}(coalesce.(df[!, embed_cols], 0.0))

    # Standardize columns
    for j in axes(X, 2)
        μ, σ = mean(X[:, j]), std(X[:, j])
        if σ > 0 && isfinite(σ)
            X[:, j] .= (X[:, j] .- μ) ./ σ
        else
            X[:, j] .= 0.0
        end
    end

    # UMAP → 2D
    Random.seed!(seed)
    metric_obj = if metric == :cosine
        CosineDist()
    elseif metric == :manhattan
        Cityblock()
    else
        Euclidean()
    end
    Y = umap(Matrix(X'), 2; n_neighbors=n_neighbors, min_dist=min_dist, metric=metric_obj)
    Y = Matrix(Y')

    canonical = [L"$\beta^+$", L"$\beta^-$", L"$\alpha$", "Stable", "Spontaneous Fission"]
    present = unique(labels_raw)
    levels = [c for c in canonical if c in present]
    append!(levels, [x for x in present if x ∉ Set(canonical)])
    decay_cat = categorical(labels_raw; levels=levels, ordered=true)
    n = length(levels)

    f = Figure(;size = (1.5inch, 1.5inch))
    ax = Axis(
        f[1, 1];
        limits = (nothing, (minimum(Y[:, 2]) - 7, nothing)),
    )
    hidedecorations!(ax)

    palette = MISTStyle.CAT_COLORS[1:n]

    h = scatter!(ax, Y[:, 1], Y[:, 2];
        color = levelcode.(decay_cat),
        colormap = palette,
        colorrange = (1, n),
        marker = :circle,
    )

    # Create legend elements
    elements = [MarkerElement(;
        marker = :circle,
        color = palette[i],
    ) for i in 1:length(levels)]

    Legend(f[1, 1], elements, levels;
        tellheight = false,
        tellwidth = false,
        orientation = :horizontal,
        halign = :right,
        valign = :bottom,
        padding=(1, 1, 1, 1),
        margin=(1, 1, 1, 1),
    )

    return f
end
