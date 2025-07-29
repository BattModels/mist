function normalize_dataset(df)
    df = transform(df, [:smi1, :smi2] => ByRow(vcat) => :compounds)
    subset!(df, "pressure [megapascal]" => ByRow(x -> abs(x - 0.1) <= 0.02))
    prop_columns = ["Density", "Molar Volume", "Molar Enthalpy"]
    rename_map = ["temperature [kelvin]" => "temperature"]
    columns = names(df)
    for col in prop_columns
        lcol = lowercase(col)
        idx = findfirst(startswith(lcol), columns)
        if !isnothing(idx)
            push!(rename_map, columns[idx] => col)
        end
        idx = findfirst(startswith("excess " * lcol), columns)
        if !isnothing(idx)
            push!(rename_map, columns[idx] => "Excess " * col)
        end
    end
    rename!(df, rename_map...)
    select!(df, ["compounds", "x1", "doi", "smi1", "smi2", last.(rename_map)...])
    subset!(df, [prop_columns..., ("Excess " .* prop_columns)...] => ByRow((x...) -> !all(ismissing.(x))))
    return df
end

function plot_mixture_coverage!(f, df)
    df = select(df,
        :smi1 => ByRow(label_functional_groups) => :fg1,
        :smi2 => ByRow(label_functional_groups) => :fg2,
    )
    fg = unique(Iterators.flatten(df.fg1))
    union!(fg, Iterators.flatten(df.fg2))
    pairs = Matrix{Int}(undef, length(fg), length(fg))
    fg_count = Vector{Int}(undef, length(fg))
    for I in CartesianIndices(pairs)
        fga  = fg[I[1]]
        fgb  = fg[I[2]]
        pairs[I] = count(zip(df.fg1, df.fg2)) do (fg1, fg2)
            return (fga in fg1 && fgb in fg2) || (fga in fg2 && fgb in fg1)
        end
        if I[1] == I[2]
            fg_count[I[1]] = count(zip(df.fg1, df.fg2)) do (fg1, fg2)
                return fga in fg1 || fga in fg2
            end
        end
    end
    sdx = sortperm(fg_count; rev=true)
    pairs = pairs[sdx, sdx]
    fg = fg[sdx]
    fg_count = fg_count[sdx]

    f = GridLayout(f)
    axb = Axis(f[1, 1];
        limits = ((0, nothing), nothing),
        xticks = WilkinsonTicks(5; k_min=2),
        yticks = (1:length(fg), titlecase.(fg)),
        # yticklabelrotation = 0.7,
        yticklabelsize=5pt,
    )

    ax = Axis(f[1, 2];
        yticksvisible=false,
        yticklabelsvisible=false,
        xticks = (1:length(fg), titlecase.(fg)),
        xticksvisible=false,
        xticklabelsvisible=false,
        backgroundcolor = :gray90
        # xticklabelrotation = 0.785
    )
    hidedecorations!(ax)
    mask = fill(NaN, length(fg), length(fg))
    tril!(mask, 0)

    h = heatmap!(ax, pairs + mask;
        colormap=MISTStyle.CONTINUOUS_COLORS,
        colorrange=(1, maximum(fg_count)),
        colorscale=log10,
        # highclip=:black,
        # lowclip=:white,
    )

    cb = Colorbar(f[0, :], h;
        label = "# Unique Mixtures",
        vertical=false,
        tellheight=true,
        tellwidth=false,
    )
    barplot!(axb, fg_count;
        direction = :x,
        strokewidth=0.25pt,
        strokecolor=:black,
        color=fg_count,
        colorscale=cb.scale,
        colormap=cb.colormap,
        highclip=cb.highclip,
        lowclip=cb.lowclip,
    )

    linkyaxes!(ax, axb)
    colgap!(f, 1, 1pt)

    return f
end
