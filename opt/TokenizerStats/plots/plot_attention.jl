using Glob
using CairoMakie
using TokenizerStats: TokenizerStats
using SmirkPaperPlots: SmirkPaperPlots, savefig
using PythonCall

const DATA_DIR = realpath(pkgdir(TokenizerStats))

function filter_model_dirs(model_ids::Vector{String}; dataset="tmQM")
    base_path = joinpath(DATA_DIR, "smirk-models")
    all_dirs = glob("**/$(dataset)", base_path)

    return filter(all_dirs) do dir
        any(id -> occursin(id, dir), model_ids)
    end
end

function load_attention_results(
    model_ids::Vector{String}, smiles::Union{String, Vector{String}};
    dataset="tmQM"
    )
    model_dirs = filter_model_dirs(model_ids; dataset=dataset)
    isempty(model_dirs) && error("No models found for IDs: $model_ids")
    println("Found $(length(model_dirs)) models")
    results = []
    smiles_list = smiles isa String ? [smiles] : smiles
    for model_path in model_dirs
        try
            data_list = [TokenizerStats.get_attention_maps(model_path, smi) for smi in smiles_list]
            if all(d -> !isnothing(d.attention_maps) && !isnothing(d.tokens), data_list)
                push!(results, (path=model_path, data=data_list))
            end
        catch e
            @warn "Error processing $model_path: $e"
        end
    end
    return results
end

function plot_attention_grid(
    smiles::String;
    model_ids::Vector{String}, dataset="tmQM",
    layer=nothing, head=nothing,
    pool="mean", colormap=:lipari,
    positions=Dict(
        "ChemBERTa v1" => (1, 1),
        "SMI-TED" => (1, 2),
        "Smirk" => (2, 1),
        )
    )

    results = load_attention_results(model_ids, smiles; dataset=dataset)

    nrows = maximum(p[1] for p in values(positions))
    ncols = maximum(p[2] for p in values(positions))
    fig = Figure(size=(230 * ncols, 190 * nrows))

    for result in results
        name = result.data[1].tokenizer_name
        haskey(positions, name) || (@warn "No position for $name"; continue)

        row, col = positions[name]
        attn = TokenizerStats.pool_pick_attention(result.data[1].attention_maps, layer, head, pool)
        attn_norm = (attn .- minimum(attn)) ./ (maximum(attn) - minimum(attn))

        ax = Axis(fig[nrows - row + 1, col], title=name)
        heatmap!(ax, attn_norm, colormap=colormap, colorrange=(0, 1))

        n = length(result.data[1].tokens)
        ax.xticks = ax.yticks = (1:n, result.data[1].tokens)
        ax.xticklabelrotation = π/2
    end

    Colorbar(fig[1:nrows, ncols + 1],
             colormap=colormap, colorrange=(0, 1),
             label="Normalized Attention",
             ticks=LinearTicks(6),
             labelsize=8,
             ticklabelsize=8,
             flip_vertical_label=true,
             tellheight=false, tellwidth=true)
    colgap!(fig.layout, 8)
    return fig
end

function plot_attention_grid(
    smiles::Vector{String};
    model_ids::Vector{String}, model_names::Vector{String},
    dataset="tmQM", layer=nothing, head=nothing,
    pool="mean", colormap=:lipari
    )

    isempty(smiles) && error("Provide at least 1 SMILES string")

    results = load_attention_results(model_ids, smiles; dataset=dataset)
    model_rows = Dict(name => idx for (idx, name) in enumerate(model_names))
    nrows = length(model_names)
    ncols = length(smiles)
    fig = Figure(size=(240 * ncols, 200 * nrows))

    for (col, smi) in enumerate(smiles)
        Label(fig[0, col], smi, fontsize=12, tellwidth=false)
    end

    for result in results
        name = result.data[1].tokenizer_name
        haskey(model_rows, name) || continue
        row = model_rows[name]

        Label(fig[row, 0], name, rotation=π/2, fontsize=14, tellheight=false)

        for (col, data) in enumerate(result.data)
            attn = TokenizerStats.pool_pick_attention(data.attention_maps, layer, head, pool)
            attn_norm = (attn .- minimum(attn)) ./ (maximum(attn) - minimum(attn))

            ax = Axis(fig[row, col]; xticklabelsize=10, yticklabelsize=10)
            heatmap!(ax, attn_norm, colormap=colormap, colorrange=(0, 1))

            n = length(data.tokens)
            ax.xticks = ax.yticks = (1:n, data.tokens)
            ax.xticklabelrotation = π/2
        end
    end

    Colorbar(fig[1:nrows, ncols + 1],
             colormap=colormap, colorrange=(0, 1),
             label="Normalized Attention",
             ticks=LinearTicks(6),
             labelsize=14,
             ticklabelsize=14,
             flip_vertical_label=true,
             tellheight=false, tellwidth=true)

    colgap!(fig.layout, 8)
    rowgap!(fig.layout, 8)
    return fig
end

function plot_delta_attention_grid(smiles::Vector{String};
                                  model_ids::Vector{String},
                                  dataset="tmQM",
                                  layer=nothing,
                                  head=nothing,
                                  pool="mean",
                                  colormap=:curl,
                                  positions=Dict(
                                      "ChemBERTa v1" => (1, 1),
                                      "SMI-TED" => (1, 2),
                                      "Smirk" => (2, 1),
                                      "SmirkGPE (MB)" => (2, 2),
                                  ))

    length(smiles) == 2 || error("Provide exactly 2 SMILES strings")

    results = load_attention_results(model_ids, smiles; dataset=dataset)

    nrows = maximum(p[1] for p in values(positions))
    ncols = maximum(p[2] for p in values(positions))
    fig = Figure(size=(220 * ncols, 180 * nrows))

    for result in results
        name = result.data[1].tokenizer_name
        haskey(positions, name) || (@warn "No position for $name"; continue)

        row, col = positions[name]

        attn1 = TokenizerStats.pool_pick_attention(result.data[1].attention_maps, layer, head, pool)
        attn2 = TokenizerStats.pool_pick_attention(result.data[2].attention_maps, layer, head, pool)

        attn1_norm = (attn1 .- minimum(attn1)) ./ (maximum(attn1) - minimum(attn1))
        attn2_norm = (attn2 .- minimum(attn2)) ./ (maximum(attn2) - minimum(attn2))
        attn_diff = attn1_norm .- attn2_norm

        ax = Axis(fig[nrows - row + 1, col], title=name)
        heatmap!(ax, attn_diff, colormap=colormap, colorrange=(-0.4, 0.4))

        n = length(result.data[1].tokens)
        tokens = replace(result.data[1].tokens, "1" => "*")
        ax.xticks = ax.yticks = (1:n, tokens)
        ax.xticklabelrotation = π/2
    end

    Colorbar(fig[1:nrows, ncols + 1],
             colormap=colormap, colorrange=(-0.4, 0.4),
             label=L"$\Delta$ Normalized Attention",
             ticks=LinearTicks(7),
             labelsize=7,
             ticklabelsize=7,
             flip_vertical_label=true,
             tellheight=false, tellwidth=true)
    colgap!(fig.layout, 8)
    return fig
end

function generate_plots(plot_fn, filename_prefix, params...;
                       layers=[nothing, 1, 2, 3, 4, 5, 6, 7],
                       heads=[nothing, 1, 3, 5, 7],
                       dataset="tmQM")
    for layer in layers, head in heads
        with_theme(SmirkPaperPlots.theme()) do
            savefig("$(filename_prefix)_l_$(layer)_h_$(head)",
                   plot_fn(params...; layer=layer, head=head, dataset=dataset))
        end
    end
end


const DEFAULT_MODEL_IDS = ["qdyxbwv3", "c1clszmm", "ti624ev1", "2scil3tk"]
const POSTIONS = Dict(
    "ChemBERTa v1" => (1, 1),
    "SMI-TED" => (1, 2),
    "Smirk" => (2, 1),
    "SmirkGPE (MB)" => (2, 2)
)

plot_comparison(model_ids=DEFAULT_MODEL_IDS;
               smiles=["Sc1cnco1", "O=Cn1cncn1", "Cn1cnc(F)n1", "Cn1cc(I)nn1"],
               dataset="tmQM") =
    generate_plots((s...; kw...) -> plot_attention_grid(s...; model_ids=model_ids,
                   model_names=["ChemBERTa v1", "Smirk", "SmirkGPE (MB)"], kw...),
                   "comparison", smiles; dataset=dataset)

plot_ambiguous(model_ids=DEFAULT_MODEL_IDS;
              smiles=["CSc1nccc([Sn](C)(C)C)n1", "FC(F)(Cl)Sn1cccc1"],
              dataset="tmQM") =
    generate_plots((s...; kw...) -> plot_attention_grid(s...; model_ids=model_ids,
                   model_names=["ChemBERTa v1", "Smirk", "SmirkGPE (MB)"], kw...),
                   "amb", smiles; dataset=dataset)

plot_single_molecule(model_ids=DEFAULT_MODEL_IDS;
                    smiles="Cl[Pt@SP1](Cl)([NH3])[NH3]",
                    positions=POSTIONS,
                    dataset="pretrained") =
    generate_plots((s; kw...) -> plot_attention_grid(s; model_ids=model_ids,
                   positions=positions, kw...),
                   "$(dataset)_$(smiles)", smiles; dataset=dataset)

plot_delta_molecules(model_ids=DEFAULT_MODEL_IDS;
                    smiles=["Cl[Pt@SP1](Cl)([NH3])[NH3]", "Cl[Pt@SP2](Cl)([NH3])[NH3]"],
                    positions=POSTIONS,
                    dataset="tmQM") =
    generate_plots((s; kw...) -> plot_delta_attention_grid(s; model_ids=model_ids,
                   positions=POSTIONS, kw...),
                   "delta", smiles; dataset=dataset, layers=[nothing, 1, 3, 5, 7])
