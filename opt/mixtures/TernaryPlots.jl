using DelaunayTriangulation
using Statistics: mean

"""
    ternary(a, b, c, values)

Plots a pseudocolor (tripcolor) plot on a ternary diagram using Delaunay triangulation
and filled triangles, similar to mpltern's tripcolor. The input vectors `a`, `b`, and `c`
represent the three compositional components, and `values` contains the scalar data to be
visualized with color. Components will be automatically normalized so that a + b + c = 1.
"""

@recipe(Ternary, a, b, c, values) do scene
    Theme(
        colormap = :viridis,
        colorrange = Makie.automatic,
        show_triangle = true,
        triangle_color = :black,
        triangle_linewidth = 2,
        show_grid = false,
        grid_color = (:gray, 0.3),
        grid_linestyle = :dash,
        grid_steps = 5,
        label_a = "A",
        label_b = "B",
        label_c = "C",
        label_fontsize = 16,
        label_offset = 25,
        show_axis_labels = true,
        axis_label_fontsize = 12,
        axis_label_offset = 15,
        axis_ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        show_ticks = true,
        tick_fontsize = 10,
        interpolate = true,  # If true, interpolate colors across triangles
    )
end

function Makie.plot!(plt::Ternary)
    # Get the input data
    a = plt.a[]
    b = plt.b[]
    c = plt.c[]
    values = plt.values[]

    # Normalize so a + b + c = 1
    total = a .+ b .+ c
    a_norm = a ./ total
    b_norm = b ./ total
    c_norm = c ./ total

    # Convert ternary coordinates to Cartesian
    x = @. 0.5 * (2 * b_norm + c_norm)
    y = @. (√3 / 2) * c_norm

    # Perform Delaunay triangulation
    points = [Point2(xi, yi) for (xi, yi) in zip(x, y)]
    tri = triangulate(points)

    # Determine color range
    crange = plt.colorrange[] === Makie.automatic ? extrema(values) : plt.colorrange[]

    # Plot each triangle
    for triangle in each_solid_triangle(tri)
        i, j, k = triangle_vertices(triangle)

        # Triangle vertices
        tri_points = [points[i], points[j], points[k]]

        if plt.interpolate[]
            # Use vertex colors for interpolation
            tri_colors = [values[i], values[j], values[k]]
            # Note: Makie's poly doesn't support per-vertex colors directly
            # Use mean color as approximation
            tri_value = mean(tri_colors)
        else
            # Use mean value
            tri_value = mean([values[i], values[j], values[k]])
        end

        # Plot filled triangle
        poly!(plt, tri_points,
              color = tri_value,
              colormap = plt.colormap,
              colorrange = crange,
              strokewidth = 0)
    end

    # Plot the triangle boundary if requested
    if plt.show_triangle[]
        corners = [Point2(0.0, 0.0), Point2(1.0, 0.0), Point2(0.5, √3/2), Point2(0.0, 0.0)]
        lines!(plt, corners,
               color = plt.triangle_color,
               linewidth = plt.triangle_linewidth)
    end

    # Add grid lines if requested
    if plt.show_grid[]
        n_steps = plt.grid_steps[]
        for i in 1:(n_steps-1)
            val = i / n_steps

            # Lines parallel to bottom edge (constant C)
            p1 = Point2(0.5 * (2 * (1-val) + 0), (√3/2) * val)
            p2 = Point2(0.5 * (2 * 0 + val), (√3/2) * val)
            lines!(plt, [p1, p2],
                   color = plt.grid_color,
                   linestyle = plt.grid_linestyle)

            # Lines parallel to left edge (constant B)
            p1 = Point2(0.5 * (2 * val + 0), (√3/2) * 0)
            p2 = Point2(0.5 * (2 * val + (1-val)), (√3/2) * (1-val))
            lines!(plt, [p1, p2],
                   color = plt.grid_color,
                   linestyle = plt.grid_linestyle)

            # Lines parallel to right edge (constant A)
            p1 = Point2(0.5 * (2 * 0 + val), (√3/2) * val)
            p2 = Point2(0.5 * (2 * (1-val) + val), (√3/2) * val)
            lines!(plt, [p1, p2],
                   color = plt.grid_color,
                   linestyle = plt.grid_linestyle)
        end
    end

    # Add corner labels
    text!(plt, 0.0, 0.0,
          text = plt.label_a[],
          align = (:center, :top),
          offset = (0, -plt.label_offset[]),
          fontsize = plt.label_fontsize[])

    text!(plt, 1.0, 0.0,
          text = plt.label_b[],
          align = (:center, :top),
          offset = (0, -plt.label_offset[]),
          fontsize = plt.label_fontsize[])

    text!(plt, 0.5, √3/2,
          text = plt.label_c[],
          align = (:center, :bottom),
          offset = (0, plt.label_offset[]),
          fontsize = plt.label_fontsize[])

    # Add tick labels along each axis if requested
    if plt.show_ticks[]
        ticks = plt.axis_ticks[]
        tick_fs = plt.tick_fontsize[]

        for t in ticks[2:end-1]  # Skip 0 and 1 (corners)
            # Bottom edge ticks (A to B, showing B values)
            x_bottom = t
            y_bottom = 0.0
            text!(plt, x_bottom, y_bottom,
                  text = string(round(t, digits=1)),
                  align = (:center, :top),
                  offset = (0, -5),
                  fontsize = tick_fs)

            # Left edge ticks (A to C, showing C values)
            x_left, y_left = ternary_to_cartesian(1-t, 0, t)
            text!(plt, x_left, y_left,
                  text = string(round(t, digits=1)),
                  align = (:right, :center),
                  offset = (-5, 0),
                  fontsize = tick_fs)

            # Right edge ticks (B to C, showing A values)
            x_right, y_right = ternary_to_cartesian(1-t, t, 0)
            text!(plt, x_right, y_right,
                  text = string(round(t, digits=1)),
                  align = (:left, :center),
                  offset = (5, 0),
                  fontsize = tick_fs)
        end
    end

    return plt
end

# Helper function for coordinate conversion (used for tick placement)
function ternary_to_cartesian(a, b, c)
    x = 0.5 * (2b + c)
    y = (√3 / 2) * c
    return x, y
end
