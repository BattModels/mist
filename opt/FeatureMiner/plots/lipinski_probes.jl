#!/usr/bin/env -S uv run julia --project=@script --startup-file=no
using Makie
using DataFrames
using GLM
using StatsBase
using JLD2: load
using JLD2: jldsave
using CategoricalArrays: categorical, levelcode
using LinearAlgebra: norm, I
using RegressionTables: regtable, LatexTable

using FeatureMiner
using MISTStyle

function load_probes(paths::Vector{String}; base)
    frames = map(paths) do path
        if isdir(path)
            probes, _ = FeatureMiner.load_fitted_probes(path)
            probes.encoder_id .= base
            probes.encoder_dataset .= basename(dirname(path))
            return probes
        end
        return DataFrame(load(path, "probes"))
    end
    df = vcat(frames...; cols=:union)
    df.encoder_id = categorical(df.encoder_id)
    ordered = ["pretrained", "tmQM"]
    dataset_order = setdiff(unique(df.encoder_dataset), ordered)
    df.encoder_dataset = categorical(df.encoder_dataset;
        levels=vcat(ordered, dataset_order),
        ordered=true,
    )
    df.location = categorical(df.location)
    sort!(df, :encoder_dataset)
    return df
end

function figure_probe_similarity(df::DataFrame; model="dh61satti", dataset="pretrained", location="output")
    df = subset(df,
        :encoder_id => ByRow(==(model)),
        :location => ByRow(==(location)),
        skipmissing=true
    )
    dropmissing!(df)
    f = Figure()


    feature_name = [
        "Lipinski",
        "H-Donor",
        "H-Acceptor",
        "MWT",
        "LogP",
    ]

    cb = Colorbar(f[1:5, 1+length(levels(df.encoder_dataset))];
        label="Cosine Similarity",
        colorrange=(0, 1),
    )

    for (ddx, dataset) in enumerate(unique(df.encoder_dataset))
        dfd = subset(df, :encoder_dataset => ByRow(==(dataset)))
        for fdx in 1:5
            ax = Axis(f[fdx, ddx])
            if fdx == 1
                ax.title = dataset
            end
            hidedecorations!(ax)
            if ddx == 1
                ax.ylabel = feature_name[fdx]
                ax.ylabelvisible = true
            end

            probe_similarity = FeatureMiner.layerwise_similarity(dfd.weight, fdx)
            heatmap!(ax, probe_similarity; MISTStyle.cb_attrs(cb, Heatmap)...)
        end
    end
    return f
end

function figure_lipinski_probes(df::DataFrame; model, label)
    f = Figure(; size=(95, 122), figure_padding=(2, 4, 2, 2))

    # Select the best probe per location
    df = combine(groupby(df, [:encoder_dataset, :location, :layer])) do gdf
        sort!(gdf, :val_loss)
        return gdf[1, :]
    end

    df = dropmissing(df)
    subset!(df,
        :encoder_id => ByRow(==(model)),
        :location => ByRow(in(["output", "intermediate", "attention"])),
    )

    # Mean AUROC
    df_auroc = combine(groupby(df, :encoder_dataset)) do gdf
        unstack(gdf[:, [:layer, :auroc]], :layer, :auroc; combine=minimum)
    end
    auroc = Matrix(df_auroc[:, 2:end])
    @info extrema(auroc)

    # Feature alignment
    gl = GridLayout(f[1,1])
    ax = Axis(gl[2, 1];
        yticks=MISTStyle.categorical_ticks(df.encoder_dataset),
        ylabel=label * " Variant Probed",
        xticks=2:2:maximum(df.layer),
    )

    h = heatmap!(ax, auroc';
        colorscale=Makie.logit,
        colorrange=(0.9, 0.995),
    )
    cb = Colorbar(gl[1, 1], h;
        label="AUROC",
        size=6pt,
        vertical=false,
        # flipaxis=false,
        ticks=[0.9, 0.99],
        tickformat="{:.0%}",
        minorticks=IntervalsBetween(5),
        minorticksvisible=true,
        labelsize=7pt,
    )

    # Additive Features
    df.additive_features = FeatureMiner.additive_features.(df.weight)
    df_af = combine(groupby(subset!(df, :location => ByRow(==("output"))), :encoder_dataset)) do gdf
        gdf = select(gdf, [:layer, :additive_features])
        unstack(gdf, :layer, :additive_features)
    end
    af = Matrix(df_af[:, 2:end])
    ax_add = Axis(gl[2, 2];
        yticks=MISTStyle.categorical_ticks(df.encoder_dataset),
        xticks=ax.xticks,
        yticklabelsvisible=false,
        yticksvisible=false,
    )
    h = heatmap!(ax_add, af')
    Colorbar(gl[1, 2], h;
        ticks=WilkinsonTicks(3),
        minorticks=IntervalsBetween(5),
        minorticksvisible=true,
        size=cb.size,
        label="Additivity",
        vertical=false,
    )
    @info extrema(af)
    Label(gl[end+1, :], "Encoder Layer")

    return f
end

function lipinski_fixed_effect(df; output="lipinski.tex")
    contrasts = Dict(
        :encoder_dataset => EffectsCoding(; base="pretrained"),
        :location => EffectsCoding(; base="output"),
    )
    m_dataset = glm(
        @formula( auroc ~ 1 + encoder_dataset), df, Normal(), LogitLink();
        contrasts,
    )
    m_layer = glm(
        @formula( auroc ~ 1 + layer), df, Normal(), LogitLink();
        contrasts,
    )
    m_location = glm(
        @formula( auroc ~ 1 + location), df, Normal(), LogitLink();
        contrasts,
    )
    m_all = glm(
        @formula( auroc ~ 1 + encoder_dataset + location + layer), df, Normal(), LogitLink();
        contrasts,
    )
    open(output, "w") do fid
        write(fid, regtable(m_all, m_layer, m_dataset, m_location;
            regression_statistics = [
                Int∘nobs => "N",
                bic => "BIC",
                Int∘dof_residual => "Resid. DoF",
                (m -> adjr2(m, :devianceratio)) => "Adj. R2",
                (m -> rmsd(response(m), predict(m))) => "RMSD",
            ],
            render=LatexTable(),

        ) |> string)
    end
    return m_all
end

function (@main)(args)
    base = args[findfirst(==("--base"), args) + 1]
    output = args[findfirst(==("--output-dir"), args) + 1]
    paths = args[findfirst(==("--probes"), args) + 1:end]
    df = load_probes(paths; base)
    label = base == "mist-28m" ? "MIST-28M" : "MIST-1.8B"
    mkpath(output)
    jldsave(joinpath(output, "probes.jld2"); probes=df)
    lipinski_fixed_effect(df; output=joinpath(output, "lipinski.tex"))
    with_theme(MISTStyle.theme()) do
        fig = figure_lipinski_probes(copy(df); model=base, label)
        MISTStyle.savefig("lipinski_linear_probes", fig; fig_dir=output)
        for location in unique(df.location)
            fig = figure_probe_similarity(copy(df); model=base, location=location)
            MISTStyle.savefig("lipinski-probe-similarity-$base-$location", fig; fig_dir=output)
        end
    end
end
