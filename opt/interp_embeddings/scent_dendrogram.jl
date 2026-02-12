#!/usr/bin/env julia

using CSV
using DataFrames
using Statistics
using LinearAlgebra
using Clustering
using CairoMakie
using Colors


function load_logits(csv_path::AbstractString)
    df = CSV.read(csv_path, DataFrame)
    @assert names(df)[1] == "smiles" "First column must be 'smiles'."

    cols = names(df)
    end_idx = (!isempty(cols) && last(cols) == "function_group") ? lastindex(cols) - 1 : lastindex(cols)
    scent_names = String.(cols[2:end_idx])
    X = Matrix{Float64}(df[:, scent_names])

    return df, X, scent_names
end


function corr_distance_scents(X::AbstractMatrix)
    n_scents = size(X, 2)
    D = zeros(Float64, n_scents, n_scents)

    for i in 1:n_scents, j in (i + 1):n_scents
        corr_val = cor(X[:, i], X[:, j])
        distance = 1.0 - (isfinite(corr_val) ? corr_val : 0.0)
        D[i, j] = D[j, i] = distance
    end

    return D
end

@inline function get_child_indices(H::Hclust, i::Int, n::Int)
    c1 = H.merge[i, 1] < 0 ? -H.merge[i, 1] : n + H.merge[i, 1]
    c2 = H.merge[i, 2] < 0 ? -H.merge[i, 2] : n + H.merge[i, 2]
    return c1, c2
end

@inline polar_to_cart(r::Real, theta::Real) = (r * cos(theta), r * sin(theta))

@inline get_node_color(node_colors, idx, default_color) =
    node_colors === nothing ? default_color : node_colors[idx]

function init_leaf_positions!(span_start, span_end, node_angles, node_radii, H, leaf_radius::Real)
    n = size(H.merge, 1) + 1
    ord = H.order
    invord = similar(ord)
    invord[ord] = 1:n

    for leaf in 1:n
        k = invord[leaf]
        span_start[leaf] = span_end[leaf] = k
        node_angles[leaf] = 2 * pi * (k - 1) / n
        node_radii[leaf] = leaf_radius
    end
    return nothing
end

function calc_internal_positions!(span_start, span_end, node_angles, node_radii, H, leaf_radius::Real, center_size::Real)
    n = size(H.merge, 1) + 1
    hmax = maximum(H.height)

    for i in 1:(n - 1)
        node_idx = n + i
        c1_idx, c2_idx = get_child_indices(H, i, n)

        span_start[node_idx] = min(span_start[c1_idx], span_start[c2_idx])
        span_end[node_idx] = max(span_end[c1_idx], span_end[c2_idx])

        mid_pos = (span_start[node_idx] + span_end[node_idx]) / 2
        node_angles[node_idx] = 2 * pi * (mid_pos - 1) / n
        node_radii[node_idx] = center_size + (leaf_radius - center_size) * (1.0 - H.height[i] / hmax)
    end
    return nothing
end

function draw_tree!(ax, H, node_angles, node_radii, node_colors, default_linecolor, lw::Real, arc_samples::Int)
    n = size(H.merge, 1) + 1

    for i in 1:(n - 1)
        node_idx = n + i
        c1_idx, c2_idx = get_child_indices(H, i, n)

        theta_p, rp = node_angles[node_idx], node_radii[node_idx]
        theta_1, r1 = node_angles[c1_idx], node_radii[c1_idx]
        theta_2, r2 = node_angles[c2_idx], node_radii[c2_idx]

        # Draw radial arc
        theta_start, theta_end = minmax(theta_1, theta_2)
        theta_end - theta_start > pi && ((theta_start, theta_end) = (theta_end, theta_start + 2 * pi))

        theta_arc = range(theta_start, theta_end, length = arc_samples)
        x_arc, y_arc = rp .* cos.(theta_arc), rp .* sin.(theta_arc)
        arc_color = get_node_color(node_colors, node_idx, default_linecolor)
        lines!(ax, x_arc, y_arc; color = arc_color, linewidth = lw)

        # Draw spokes to children
        for (theta, r, c_idx) in [(theta_1, r1, c1_idx), (theta_2, r2, c2_idx)]
            x_arc, y_arc = polar_to_cart(rp, theta)
            x_child, y_child = polar_to_cart(r, theta)
            col = get_node_color(node_colors, c_idx, default_linecolor)
            lines!(ax, [x_arc, x_child], [y_arc, y_child]; color = col, linewidth = lw)
        end
    end
    return nothing
end

function add_leaf_labels!(ax, node_angles, leaf_names, leaf_colors, label_radius::Real, label_size::Real)
    n = length(leaf_names)

    for leaf in 1:n
        theta = node_angles[leaf]
        theta_mod = mod(theta, 2 * pi)
        rotation = pi / 2 < theta_mod < 3 * pi / 2 ? theta + pi : theta
        align = pi / 2 < theta_mod < 3 * pi / 2 ? (:right, :center) : (:left, :center)
        col = get_node_color(leaf_colors, leaf, :black)
        x, y = polar_to_cart(label_radius, theta)

        text!(ax, x, y;
            text = titlecase(replace(leaf_names[leaf], " " => "")),
            rotation = rotation,
            align = align,
            color = col,
            font = "Helvetica",
            fontsize = label_size)
    end
    return nothing
end


function radial_dendrogram!(ax, H::Hclust;
    leaf_names::Vector{<:AbstractString},
    leaf_colors::Union{Nothing, AbstractVector{<:Colorant}} = nothing,
    node_colors::Union{Nothing, AbstractVector{<:Colorant}} = nothing,
    lw::Real = 1.5,
    default_linecolor = :black,
    label_radius::Real = 0.9,
    label_size::Real = 8,
    center_size::Real = 0.05,
    arc_samples::Int = 50,
)
    n = size(H.merge, 1) + 1
    @assert length(leaf_names) == n

    span_start, span_end = fill(0, 2n - 1), fill(0, 2n - 1)
    node_angles, node_radii = zeros(Float64, 2n - 1), zeros(Float64, 2n - 1)

    leaf_radius = 0.022
    init_leaf_positions!(span_start, span_end, node_angles, node_radii, H, leaf_radius)
    calc_internal_positions!(span_start, span_end, node_angles, node_radii, H, leaf_radius, center_size)

    draw_tree!(ax, H, node_angles, node_radii, node_colors, default_linecolor, lw, arc_samples)
    add_leaf_labels!(ax, node_angles, leaf_names, leaf_colors, label_radius, label_size)

    return nothing
end


function plot_radial_dendrogram(D::AbstractMatrix, scent_names::Vector{<:AbstractString};
    linkage::Symbol = :average, k::Union{Nothing, Int} = nothing, figsize = (500, 500))
    @assert size(D, 1) == size(D, 2) && length(scent_names) == size(D, 1)
    n = size(D, 1)

    H = hclust(D; linkage = linkage)
    leaf_colors, node_colors = nothing, nothing

    if k !== nothing
        lids = cutree(H, k = k)

        palette = to_colormap(:tab10, 13)
        cmap = Dict(cid => palette[mod1(i, length(palette))] for (i, cid) in enumerate(unique(lids)))
        leaf_colors = [cmap[c] for c in lids]

        # Propagate leaf colors up the tree
        node_colors = fill(RGBAf(0.5, 0.5, 0.5, 1.0), 2n - 1)
        node_colors[1:n] .= leaf_colors

        for i in 1:(n - 1)
            node_idx = n + i
            c1_idx, c2_idx = get_child_indices(H, i, n)
            node_colors[node_idx] = node_colors[c1_idx] == node_colors[c2_idx] ?
                                    node_colors[c1_idx] : RGBAf(0.3, 0.3, 0.3, 1.0)
        end
    end

    fig = Figure(size = figsize, figure_padding = 10)
    lim = 0.03
    ax = Axis(fig[1, 1]; limits = ((-lim, lim), (-lim, lim)))
    hidespines!(ax)
    hidedecorations!(ax)

    radial_dendrogram!(ax, H;
        leaf_names = scent_names,
        leaf_colors = leaf_colors,
        node_colors = node_colors,
        lw = 1.5,
        default_linecolor = :black,
        label_radius = 0.0225,
        label_size = 10,
        center_size = 0.004)

    return fig, ax, H, leaf_colors
end

csv_path = "raw_prediction.csv"
df, X, scent_names = load_logits(csv_path)
D = corr_distance_scents(X)
fig, ax, H, colors = plot_radial_dendrogram(D, scent_names; linkage = :complete, k = 12, figsize = (510, 500))
outpath = replace(csv_path, r"\.csv$" => "_radial_dendrogram.pdf")
save(outpath, fig)
