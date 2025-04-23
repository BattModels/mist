using Makie
using DataFrames
using GLM
using StatsBase
using JLD2: jldopen
using CategoricalArrays: categorical, levelcode
using LinearAlgebra: norm, I
using RegressionTables: regtable, LatexTable

using FeatureMiner
using MISTStyle

function load_probes()
    df =  DataFrame(jldopen("lipinski_probes_ti624ev1.jld2")["probes"])
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

function figure_probe_similarity(df::DataFrame, model="dh61satti", dataset="toxcast", location="output")
    df = subset(df,
        :encoder => ByRow(==(model)),
        :dataset => ByRow(==(dataset)),
        :location => ByRow(==(location)),
        skipmissing=true
    )
    dropmissing!(df)
    f = Figure()

    for (fdx, feat) in enumerate(unique(df.feature_id))
        @show df_f = subset(df, :feature_id => ByRow(==(feat)))
        df_f = first(eachrow(df_f))
        ax = Axis(f[fdx, 1]; title=df_f.id)
        heatmap!(ax, df_f.probe_similarity)
    end
    return f
end

function figure_lipinski_probes(df::DataFrame)
    f = Figure(size=(2inch, 1inch))

    df = dropmissing(df)
    subset!(df,
        :encoder_id => ByRow(==("ti624ev1")),
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
    ax = Axis(gl[1, 1];
        xticks=categorical_ticks(df.encoder_dataset),
        yticks=2:2:8,
        # limits=(nothing, ),
        xticklabelsvisible=false,
        xticksvisible=false,
    )

    h = heatmap!(ax, auroc)
    Colorbar(gl[1, 2], h; label="AUROC")

    # Additive Features
    df.additive_features = FeatureMiner.additive_features.(df.weight)
    df_af = combine(groupby(subset!(df, :location => ByRow(==("output"))), :encoder_dataset)) do gdf
        unstack(gdf[:, [:layer, :additive_features]], :layer, :additive_features)
    end
    af = Matrix(df_af[:, 2:end])
    ax_add = Axis(gl[2, 1];
        xticks=categorical_ticks(df.encoder_dataset),
        yticks=ax.yticks,
    )
    h = heatmap!(ax_add, af; colorrange=(0, 1))
    Colorbar(gl[2, 2], h; label="Additivity")
    @info extrema(af)

    # Add single Y-axis label
    Label(gl[:, 0], text = "Encoder Layer", rotation = pi/2)

    return f
end

function lipinski_fixed_effect(df)
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
    display(m)
    open("lipinski.tex", "w") do fid
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
    return m
end
