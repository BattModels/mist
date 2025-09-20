using Makie
using MISTStyle
using JSON: JSON
using DataFrames: DataFrame, ByRow, subset, combine, groupby
using CategoricalArrays: categorical, levelcode, levels

const HARTREE_TO_EV = 27.211_386_245_981

function load_results(filename)
    df = DataFrame(JSON.parsefile(filename))
    df = combine(groupby(df, [:metal, :organic])) do gdf
        first(gdf.metal) == "organic" ? gdf[[1], :] : gdf
    end
    df.metal = categorical(df.metal)
    df.organic = categorical(df.organic;
        levels=["benzene", "naphthalene", "pyrene", "coronene"],
    )
    df.HL_Gap .*= HARTREE_TO_EV
    df.HOMO_Energy .*= HARTREE_TO_EV
    df.LUMO_Energy .*= HARTREE_TO_EV
    return df
end

function plot_organic!(f, df::DataFrame, organic; kwargs...)
    if organic isa AbstractString
        organic = organic => uppercasefirst(organic)
    end
    organic, title = split_label(organic)
    df = subset(df, :organic => ByRow(x -> x == organic))
    kwargs = merge((; title), kwargs)
    return plot_organic!(f, df; kwargs...)
end

split_label(channel::AbstractString) = channel, channel
split_label(chn_label::Pair) = chn_label[1], chn_label[2]

function plot_organic!(f, df::DataFrame; channel = "HL_Gap", title=nothing, limits=nothing)
    chn, ylabel = split_label(channel)
    ax = Axis(f;
        title,
        limits,
        xlabel = "Metal Formal Charge",
        yticks = WilkinsonTicks(3),
        ylabel,
    )

    h = barplot!(ax, df[!, :charge], df[!, chn];
        dodge = levelcode.(df[!, :metal]),
        color = levelcode.(df[!, :metal]),
        colormap = MISTStyle.CAT_COLORS,
        colorrange = (1, 10),
    )
    return h, ax
end

DEFAULT_CHANNELS = [
    "Dipole_M" => L"$$Dipole Moment\n(D)",
    "Polarizability" => L"Polarizability\n($\AA$)",
    "HOMO_Energy" => L"$$HOMO\n(eV)",
    "LUMO_Energy" => L"$$LUMO\n(eV)",
    "HL_Gap" => L"$$Gap\n(eV)",
]

function plot_results(df, channels=DEFAULT_CHANNELS)
    f = Figure(; size=(4inch, 3inch))

    local h
    organics = levels(df.organic)
    plt_axes = Matrix{Any}(undef, length(channels), length(organics))
    for (cdx, channel) in enumerate(channels)
        chn, _ = split_label(channel)
        l, u = extrema(df[!, chn])
        if 0 <= l && 0 <= u
            limits = (nothing, (0.0, nothing))
        elseif l <= 0 && u <= 0
            limits = (nothing, (floor(l), 0.0))
        else
            limits = (nothing, nothing)
        end
        for (odx, organic) in enumerate(organics)
            h, ax = plot_organic!(f[cdx, odx], df, organic; channel, limits)
            if cdx != length(channels)
                hidexdecorations!(ax, grid=false)
            else
                ax.xlabelvisible[] = false
            end
            if cdx != 1
                ax.titlevisible[] = false
            end
            if odx != 1
                hideydecorations!(ax, grid=false)
            end
            plt_axes[cdx, odx] = ax
        end
    end
    for i in axes(plt_axes, 1)
        linkxaxes!(plt_axes[i, :]...)
    end
    for i in axes(plt_axes, 2)
        linkyaxes!(plt_axes[i, :]...)
    end
    Label(f[end+1,:], "Formal Metal Charge")

    elems = map(enumerate(levels(df.metal))) do (color, label)
        PolyElement(; label, color, colormap=h.colormap, colorrange=h.colorrange)
    end
    elems[end].label[] = "PAH"
    Legend(f[0, :], elems, MISTStyle.label.(elems);
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
    )

    rowgap!(f.layout, 6pt)
    resize_to_layout!(f)

    return f
end

function plot_all()
    files = [
        "results_auto.json" => "auto",
        "results_auto6.json" => "auto6",
        "results_eta2.json" => "eta2",
        "results_eta6.json" => "eta6",
        "results_center2.json" => "center2",
    ]
    for (file, label) in files
        @info file
        df = load_results(file)
        for loc in ["inner", "outer", "min"]
            # Select location based result
            if loc == "inner"
                dfp = subset(df, :placement => ByRow(!=("outer")))
            elseif loc == "outer"
                dfp = subset(df, :placement => ByRow(!=("inner")))
            elseif loc == "min"
                dfp = combine(groupby(df, [:organic, :metal, :charge])) do gdf
                    idx = argmin(gdf[!, :Electronic_E])
                    return gdf[idx, :]
                end
            end

            with_theme(MISTStyle.theme()) do
                plot_results(df)
            end |> MISTStyle.savefig("metallic_pah_$(label)_$(loc)")
        end
    end
    return nothing
end
