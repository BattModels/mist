using Makie
using MISTStyle
using PythonCall
using DataFrames
using CSV: CSV
using StatsBase: mean
using Mixtures

function plot_dataset_sparity!(f, df, prop_columns)
    ax = Axis(f[1, 1];
        limits = (nothing, (1, nothing)),
        xticks = (1:length(prop_columns), prop_columns),
        xticklabelrotation=0.2,
        ylabel="# of Unique Mixtures",
        yscale=log10,
    )
    x = []
    dodge = Int[]
    y = []
    for (idx, col) in enumerate(prop_columns)
        push!(y, count(!ismissing, df[!, col]))
        push!(x, idx)
        push!(dodge, 1)
        push!(y, count(!ismissing, df[!, "Excess $(col)"]))
        push!(x, idx)
        push!(dodge, 2)
    end
    barplot!(ax, x, y;
        dodge,
        color=dodge,
        colormap=MISTStyle.CAT_COLORS,
        colorrange=(1, length(MISTStyle.CAT_COLORS)),
        fillto=0.5,
    )
    elems = [
        PolyElement(; color=MISTStyle.CAT_COLORS[1], label="Total"),
        PolyElement(; color=MISTStyle.CAT_COLORS[2], label="Excess"),
    ]
    Legend(f[1, 1], elems, MISTStyle.label.(elems);
        valign=:top,
        halign=:right,
        tellwidth=false,
        tellheight=false,
    )
    return f


end

function plot_mixture_properties!(f, df, prop_columns)
    f = GridLayout(f)
    hist_kwargs = (;
        normalization=:none,
        bins=40,
        fillto=0.1,
    )
    ax_kwargs = (;
        limits = (nothing, (0.1, 1e5)),
        yticksvisible=false,
        yscale=log10,
        yticklabelsvisible=false,
    )
    axes = Axis[]
    ax = Axis(f[1, 1]; xlabel="Temperature (K)", ax_kwargs...)
    push!(axes, ax)
    hist!(ax, df[!, "temperature"]; hist_kwargs...)

    ax = Axis(f[2, 1]; xlabel="Excess Molar Enthalpy (J/mol)", ax_kwargs...)
    push!(axes, ax)
    hist!(ax, collect(skipmissing(df[!, "Excess Molar Enthalpy"])); hist_kwargs...)

    # Excess Properties Distributions
    gl_excess = GridLayout(f[:, 2:3])
    tfs = map(prop_columns) do col
        ["Excess $(col)", col] => ByRow(/) => "Percent Excess $(col)"
    end
    df = transform(df, tfs)
    for (idx, (col, units)) in enumerate([("Density", L"g/\mathrm{mol}^3"), ("Molar Volume", L"cm^3/mol")])
        # Absolute Excess
        ax = Axis(gl_excess[1, idx];
            xlabel=L"Excess %$(col) ($ %$units $)",
            ax_kwargs...
        )
        push!(axes, ax)
        v = collect(skipmissing(df[!, "Excess $(col)"]))
        isempty(v) && continue
        h = hist!(ax, v; hist_kwargs...)

        # Percent Excess
        ax = Axis(gl_excess[2, idx];
            ax_kwargs...,
            xlabel="Percent Excess $(col)",
            xtickformat="{:.0%}",
        )
        push!(axes, ax)
        v = collect(skipmissing(df[!, "Percent Excess $(col)"]))
        isempty(v) && continue
        hist!(ax, v; hist_kwargs...)
    end
    return f
end

function mixture_dataset(df)
    prop_columns = ["Density", "Molar Volume", "Molar Enthalpy"]

    f = Figure(; size=(4.5inch, 3inch))
    plot_mixture_properties!(f[1, 1:2], df, prop_columns)
    Label(f[1, 1, Left()];
        text=L"$\ln$ Examples",
        rotation=pi/2,
        tellwidth=true,
        tellheight=false,
        valign=:center,
        halign=:right,
        padding=(0, 4pt, 0, 0),
    )

    # Restrict to pair-wise observations
    df_mix = combine(groupby(df, [:doi, :compounds, :temperature])) do gdf
        out = Dict{String, Any}("smi1" => first(gdf[!, "smi1"]), "smi2" => first(gdf[!, "smi2"]))
        for col in prop_columns
            out[col] = mean(skipmissing(gdf[!, col]))
            out["Excess " * col] = mean(skipmissing(gdf[!, "Excess " * col]))
        end
        return NamedTuple(Symbol(k) => (!(v isa String) && isnan(v)) ? missing : v for (k, v) in pairs(out))
    end
    Mixtures.plot_mixture_coverage!(f[2, 1], df_mix)
    plot_dataset_sparity!(f[2, 2], df_mix, prop_columns)

    MISTStyle.sublabel!(f[1, 1:2, TopLeft()], "a"; left=5)
    MISTStyle.sublabel!(f[2, 1, TopLeft()], "b"; left=5)
    MISTStyle.sublabel!(f[2, 2, TopLeft()], "c"; left=5)


    return f
end

function dataset_stats(df)
    compounds = unique(Iterators.flatten([df.smi1, df.smi2]))
    @info "Number of compounds" length(compounds)

    mixtures = map(df.smi1, df.smi2) do smi1, smi2
        tuple(sort([smi1, smi2])...)
    end
    @info "Number of mixtures" length(unique(mixtures))
end

function (@main)(ARGS=[])
    filename = joinpath(dirname(@__FILE__), "..", "..", "excess_v6.csv")
    filename = length(ARGS) > 1 ? getindex(ARGS, 1) : filename
    df = Mixtures.normalize_dataset(DataFrame(CSV.File(filename)))
    dataset_stats(df)
    with_theme(MISTStyle.theme()) do
        MISTStyle.savefig("mixture_dataset", mixture_dataset(df))
    end
    return 0
end
