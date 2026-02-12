using Makie
using DesignRules
using MISTStyle
using DataFrames
using Colors
using PythonCall
using CSV: CSV
using StatsBase: mean


sigmoid(x) = 1 / (1 + exp(-x))

extract_probs(vals) = eltype(vals) <: Real ? sigmoid.(vals) : sigmoid.(mean.(vals))

get_scent_cols(df) = filter(col -> !(col in ["smi", "type", "n_carbon", "branch_pattern"]), names(df))

grid_position(i, ncols) = (div(i - 1, ncols) + 1, mod(i - 1, ncols) + 1)


function smiles_to_image(smi::String; img_size=(150, 150))
    rdkit = @pyconst(pyimport("rdkit.Chem"))
    rdkit_draw = @pyconst(pyimport("rdkit.Chem.Draw"))
    numpy = @pyconst(pyimport("numpy"))

    mol = rdkit.MolFromSmiles(smi)
    img_pil = rdkit_draw.MolToImage(mol, size=img_size)
    img_array = pyconvert(Array, numpy.array(img_pil))

    return permutedims(
        [RGB(img_array[i,j,1]/255, img_array[i,j,2]/255, img_array[i,j,3]/255)
         for i in axes(img_array,1), j in axes(img_array,2)],
        (2,1)
    )
end


function predict_scent(branched::Bool = false)
    model = DesignRules.load_huggingface_model("mist-models/mist-26.9M-48kpooqf-odour")
    df = branched ? DesignRules.branched_fragrance_compounds(5) :
                    DesignRules.fragrance_compounds(15)

    df.smi .= DesignRules.encode.(df.smi; encoding="smiles-kekule")
    unique!(df, :smi)

    df_logits = DesignRules.predict_all(df, model; n=10)
    unique!(df_logits, :smi)

    scent_cols = get_scent_cols(df_logits)
    cols_to_keep = filter(
        col -> col ∉ scent_cols || sum(sigmoid.(mean.(df_logits[!, col])) .> 0.5) > 0,
        names(df_logits)
    )

    return select(df_logits, cols_to_keep)
end


function plot_scent_by_type(df::DataFrame)
    types = sort(unique(df.type))
    scent_cols = get_scent_cols(df)
    colors = MISTStyle.CAT_COLORS
    linestyles = [
        :solid, (:dash, :dense), (:dot, :dense),
        :dashdot, :dashdotdot, (:dot, :loose), (:dash, :loose)
    ]

    scent_colors = Dict(s => colors[mod1(i, length(colors))] for (i, s) in enumerate(scent_cols))
    scent_linestyles = Dict(s => linestyles[mod1(i, length(linestyles))] for (i, s) in enumerate(scent_cols))

    fig = Figure(size=(250mm, 120mm), figure_padding=(3, 3, 3, 3))
    n_cols = ceil(Int, length(types) / 2)

    for (i, type) in enumerate(types)
        df_type = filter(row -> row.type == type, df)
        row_pos, col_pos = grid_position(i, n_cols)
        sublayout = GridLayout(fig[row_pos, col_pos])

        ax = Axis(
            sublayout[1, 1],
            xlabel="Number of Carbons",
            ylabel="Probability",
            title=type,
            tellheight=true,
            limits=((minimum(df_type.n_carbon), maximum(df_type.n_carbon)), (0, 1))
        )

        plotted_elements, plotted_labels = [], []

        for scent in scent_cols
            y_mean = mean.(df_type[!, scent])
            y_prob = sigmoid.(y_mean)

            if any(y_prob .> 0.5)
                y_std = [val.std for val in df_type[!, scent]]

                band!(
                    ax, df_type.n_carbon,
                    sigmoid.(y_mean .- y_std),
                    sigmoid.(y_mean .+ y_std),
                    color=(scent_colors[scent], 0.2)
                )
                lines!(
                    ax, df_type.n_carbon, y_prob,
                    color=scent_colors[scent],
                    linestyle=scent_linestyles[scent]
                )

                push!(
                    plotted_elements,
                    LineElement(
                        color=scent_colors[scent],
                        linestyle=scent_linestyles[scent]
                    )
                )
                push!(plotted_labels, scent)
            end
        end

        Legend(
            sublayout[2, 1], plotted_elements, plotted_labels,
            orientation=:horizontal,
            tellwidth=false,
            tellheight=true,
            nbanks=max(1, ceil(Int, length(plotted_elements) / 6)),
            patchsize=(8, 4)
        )
    end

    return fig
end


function plot_branched_scent_comparison(df::DataFrame)
    scent_cols = get_scent_cols(df)
    scents_to_plot = filter(s -> any(extract_probs(df[!, s]) .> 0.5), scent_cols)

    types = sort(unique(df.type))
    colors = MISTStyle.CAT_COLORS
    max_variants = maximum(nrow(filter(row -> row.type == type, df)) for type in types)
    scent_colors = [colors[mod1(i, length(colors))] for i in eachindex(scents_to_plot)]

    fig = Figure(size=(175mm, 140mm), figure_padding=(5, 5, 5, 5))

    for (i, type) in enumerate(types)
        df_type = filter(row -> row.type == type, df)
        row_pos, col_pos = grid_position(i, 2)

        ax = Axis(
            fig[row_pos, col_pos],
            ylabel="Probability",
            title=type,
            xticklabelrotation=π/3,
            xticklabelalign=(:right, :center),
            limits=(0.5, max_variants + 0.5, 0, 1.3),
            yticks=[0, 0.25, 0.5, 0.75, 1.0]
        )

        x_pos_scaled = collect(range(1, max_variants, length=nrow(df_type)))
        x_positions = repeat(x_pos_scaled, outer=length(scents_to_plot))
        dodge_groups = repeat(eachindex(scents_to_plot), inner=nrow(df_type))
        heights = reduce(vcat, [extract_probs(df_type[!, s]) for s in scents_to_plot])

        barplot!(
            ax, x_positions, heights,
            dodge=dodge_groups,
            color=[scent_colors[dodge_groups[j]] for j in eachindex(dodge_groups)],
            dodge_gap=0.01,
            width=0.9
        )

        ax.xticks = (x_pos_scaled, df_type.smi)

        for (x_pos, smi) in zip(x_pos_scaled, df_type.smi)
            try
                img = permutedims(smiles_to_image(smi), (2, 1))
                image!(ax, (x_pos - 0.4)..(x_pos + 0.4), 1.05..1.25, img)
            catch e
                @warn "Failed to generate image for SMILES: $smi" exception=e
            end
        end
    end

    Legend(
        fig[3, :],
        [PolyElement(color=c) for c in scent_colors],
        scents_to_plot,
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
        nbanks=max(1, ceil(Int, length(scents_to_plot) / 10))
    )

    return fig
end
