using Makie
using DesignRules
using MISTStyle
using DataFrames
using CSV: CSV
using StatsBase: mean

sigmoid(x) = 1 / (1 + exp(-x))

function predict_scent()
    models = (
        DesignRules.load_huggingface_model("mist-models/mist-26.9M-48kpooqf-odour"),
    )
    df = DesignRules.fragrance_compounds(15)
    df.smi .= DesignRules.encode.(df.smi; encoding="smiles-kekule")
    df_logits = DesignRules.predict_all(
        df,
        models...;
        n=10
    )

    # Filter columns based on sigmoid threshold
    cols_to_keep = filter(names(df_logits)) do col
        col == "smi" || col == "type" || col == "n_carbon" ||
        sum(sigmoid.(mean.(df_logits[!, col])) .> 0.5) > 0
    end
    df_logits = select(df_logits, cols_to_keep)
    return df_logits
end

function plot_scent_by_type(df::DataFrame)

    extract_probs(vals) = eltype(vals) <: Real ? sigmoid.(vals) : sigmoid.(mean.(vals))

    types = sort(unique(df.type))
    scent_cols = filter(col -> !(col in ["smi", "type", "n_carbon"]), names(df))

    scents_to_plot = filter(scent_cols) do scent
        any(types) do type
            df_type = filter(row -> row.type == type, df)
            any(extract_probs(df_type[!, scent]) .> 0.5)
        end
    end

    colors = MISTStyle.CAT_COLORS
    linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot]
    scent_colors = Dict(s => colors[mod1(i, length(colors))] for (i, s) in enumerate(scents_to_plot))
    scent_linestyles = Dict(s => linestyles[mod1(i, length(linestyles))] for (i, s) in enumerate(scents_to_plot))

    n_cols = ceil(Int, length(types) / 2)
    fig = Figure(; size = (250mm, 120mm), figure_padding = (3, 3, 3, 3))

    for (i, type) in enumerate(types)
        df_type = filter(row -> row.type == type, df)
        row_pos = div(i - 1, n_cols) + 1
        col_pos = mod(i - 1, n_cols) + 1

        sublayout = GridLayout(fig[row_pos, col_pos])

        ax = Axis(sublayout[1, 1],
            xlabel="Number of Carbons",
            ylabel="Probability",
            tellheight=true,
            limits=((minimum(df_type.n_carbon), maximum(df_type.n_carbon)), (0, 1)),
            title=type)


        plotted_elements = []
        plotted_labels = []

        for scent in scents_to_plot
            x = df_type.n_carbon
            y_vals = df_type[!, scent]
            color = scent_colors[scent]
            linestyle = scent_linestyles[scent]

            y_mean = mean.(y_vals)
            y_prob = sigmoid.(y_mean)
            if any(y_prob .> 0.5)
                y_std = [val.std for val in y_vals]
                y_lower = sigmoid.(y_mean .- y_std)
                y_upper = sigmoid.(y_mean .+ y_std)
                band!(ax, x, y_lower, y_upper, color=(color, 0.2))
                lines!(ax, x, y_prob, color=color, linestyle=linestyle, label=scent)
                scatter!(ax, x, y_prob, color=color)

                push!(plotted_elements, LineElement(color=color, linestyle=linestyle))
                push!(plotted_labels, scent)
            end
        end

        Legend(sublayout[2, 1], plotted_elements, plotted_labels,
                orientation=:horizontal, tellwidth=false, tellheight=true,
                nbanks=max(1, ceil(Int, length(plotted_elements) / 6)),
                patchsize=(8, 4))
    end

    return fig
end
