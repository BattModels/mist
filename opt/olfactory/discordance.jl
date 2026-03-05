using Makie
using MISTStyle
using PythonCall: pyimport, pyconvert, pylist, @pyconst
using DataFrames
using CSV: CSV
using JSON3
using Statistics: mean
using Distances: pairwise, Euclidean, CosineDist
using Printf: @sprintf


function kekulize(smi::String)
    Chem = pyimport("rdkit.Chem")
    mol = Chem.MolFromSmiles(smi)
    mol === nothing && return nothing
    Chem.Kekulize(mol)
    pyconvert(String, Chem.MolToSmiles(mol; kekuleSmiles=true))
end

function canonicalize(smi::String)
    Chem = pyimport("rdkit.Chem")
    mol = Chem.MolFromSmiles(smi)
    mol === nothing && return nothing
    pyconvert(String, Chem.MolToSmiles(mol))
end

function load_model(model_path="mist-models/mist-26.9M-48kpooqf-odour")
    transformers = pyimport("transformers")
    torch = pyimport("torch")
    model = transformers.AutoModel.from_pretrained(model_path; trust_remote_code=true)
    model.eval()
    device = if pyconvert(Bool, torch.cuda.is_available())
        torch.device("cuda")
    elseif pyconvert(Bool, torch.backends.mps.is_available())
        torch.device("mps")
    else
        torch.device("cpu")
    end
    model.to(device)
end

function load_code_to_smiles(label_path="../../osmo_data/safe_to_share.csv",
                              anchor_path="../../osmo_data/anchor_labels.json")
    labels = DataFrame(CSV.File(label_path))
    code_to_smiles = Dict(zip(labels[!, "Sample Identifier"], labels.SMILES))
    anchor_smiles = JSON3.read(read(anchor_path, String), Dict{String, String})
    merge!(code_to_smiles, anchor_smiles)
end

function get_olfactory_profiles(model, smiles_list::Vector{String}; batch_size=32, mc_iterations=10)
    kekule_smiles = [kekulize(s) for s in smiles_list]

    model.train()
    mc_logits = []
    channel_names = nothing

    for _ in 1:mc_iterations
        all_logits = []

        for i in 1:batch_size:length(kekule_smiles)
            batch = pylist(kekule_smiles[i:min(i+batch_size-1, end)])
            output = model.predict(batch)

            if isnothing(channel_names)
                channel_names = sort(collect(pyconvert(Vector{String}, output.keys())))
            end

            logits = reduce(hcat, [pyconvert(Vector{Float64}, output[ch]["value"]) for ch in channel_names])
            push!(all_logits, logits)
        end

        push!(mc_logits, vcat(all_logits...))
    end

    mean_logits = mean(mc_logits)
    mean_probs = @. 1 / (1 + exp(-mean_logits))
    model.eval()

    (mean_logits, mean_probs, channel_names)
end

function validate_triplet_predictions(label_path="../../osmo_data/safe_to_share.csv",
                                       anchor_path="../../osmo_data/anchor_labels.json",
                                       molecule_path="../../osmo_data/Data S5.csv")
    df = DataFrame(CSV.File(molecule_path))
    code_to_smiles = load_code_to_smiles(label_path, anchor_path)
    df.smiles = [get(code_to_smiles, code, missing) for code in df[!, "RedJade Code"]]

    dropmissing!(df, :smiles)
    select!(df, Not(:Alcoholic))
    rename!(df, :Jasmine => :jasmin)
    rename!(df, [col => lowercase(string(col)) for col in names(df)])

    model = load_model()
    all_logits, all_probs, channel_names = get_olfactory_profiles(model, df.smiles)

    mist_pred_full = DataFrame(all_logits, channel_names)
    label_cols = names(df)[2:end-1]
    mist_pred = mist_pred_full[:, label_cols]

    for pred_df in (mist_pred_full, mist_pred)
        pred_df.smiles = [kekulize(s) for s in df.smiles]
    end
    df.smiles = [kekulize(s) for s in df.smiles]

    (mist_pred_full, mist_pred, df[:, 2:end])
end

function load_triplets(triplets)
    anchors = unique(triplets[triplets.roleA .== "anchor", "RedJade Code A"])
    triples = []

    for anchor in anchors
        mask_anchor = (triplets.roleA .== "anchor") .& (triplets[!, "RedJade Code A"] .== anchor)

        label_cliffs = unique(triplets[mask_anchor .& (triplets.roleB .== "label_cliff"), "RedJade Code B"])
        struct_cliffs = unique(triplets[mask_anchor .& (triplets.roleB .== "struct_cliff"), "RedJade Code B"])

        for l in label_cliffs, s in struct_cliffs
            has_struct = any((triplets.roleA .== "anchor") .&
                            (triplets[!, "RedJade Code A"] .== anchor) .&
                            (triplets[!, "RedJade Code B"] .== s))
            has_label = any((triplets.roleA .== "anchor") .&
                           (triplets[!, "RedJade Code A"] .== anchor) .&
                           (triplets[!, "RedJade Code B"] .== l))

            if has_struct && has_label
                push!(triples, (anchor, s, l))
            end
        end
    end

    @info "$(length(triples)) triples loaded"
    triples
end

function compute_distance_metrics(triplets_mean, code_to_smiles, mist_pred_full, mist_pred, gnn_pred)

    pred_dfs = (mist_pred_full, mist_pred, gnn_pred)
    indices = [Dict(zip(df.smiles, 1:nrow(df))) for df in pred_dfs]
    matrices = [Matrix(select(df, Not(:smiles))) for df in pred_dfs]

    distances = (
        pairwise(CosineDist(), matrices[2], dims=1),
        pairwise(Euclidean(), matrices[2], dims=1),
        pairwise(CosineDist(), matrices[1], dims=1),
        pairwise(CosineDist(), matrices[3], dims=1)
    )

    metric_names = ["Cosine Distance (MIST)", "Euclidean Distance (MIST)",
                    "Cosine Distance (MIST Full)", "Predicted Label Distance (GNN)"]

    for (dist_mat, name, idx_dict) in zip(distances, metric_names, [indices[2], indices[2], indices[1], indices[3]])
        triplets_mean[!, name] = [
            dist_mat[idx_dict[code_to_smiles[row["RedJade Code A"]]],
                     idx_dict[code_to_smiles[row["RedJade Code B"]]]]
            for row in eachrow(triplets_mean)
        ]
    end

    triplets_mean
end

function build_triples_dataframe(triples, triplets_mean, distances)
    triples_data = []

    for (anchor, struct_cliff, label_cliff) in triples, metric in distances,
        (smi, cliff_type) in [(struct_cliff, "struct"), (label_cliff, "label")]

        idx = findfirst(r -> r["RedJade Code A"] == anchor &&
                            r["RedJade Code B"] == smi &&
                            r.roleB == "$(cliff_type)_cliff", eachrow(triplets_mean))

        !isnothing(idx) && push!(triples_data,
            (anchor=anchor, struct_cliff=struct_cliff, label=label_cliff,
             metric=metric, cliff=cliff_type, value=triplets_mean[idx, metric]))
    end

    DataFrame(triples_data)
end

function plot_metric_panel!(ax, metric_data, metric)
    struct_data = filter(r -> r.cliff == "struct", metric_data).value
    label_data = filter(r -> r.cliff == "label", metric_data).value

    if !isempty(struct_data)
        violin!(ax, fill(0.0, length(struct_data)), struct_data;
                color=MISTStyle.UM_COLORS.maize,
                alpha = 0.8,
                width=0.8,
                datalimits=extrema)
    end
    if !isempty(label_data)
        violin!(ax, fill(1.0, length(label_data)), label_data;
                color=MISTStyle.UM_COLORS.blue,
                alpha = 0.8,
                width=0.8,
                datalimits=extrema)
    end

    discordant_count = 0
    total_count = 0

    for triple in unique(zip(metric_data.anchor, metric_data.struct_cliff, metric_data.label))
        triple_data = filter(r -> r.anchor == triple[1] &&
                                  r.struct_cliff == triple[2] &&
                                  r.label == triple[3], metric_data)

        struct_vals = filter(r -> r.cliff == "struct", triple_data).value
        label_vals = filter(r -> r.cliff == "label", triple_data).value

        (isempty(struct_vals) || isempty(label_vals)) && continue

        struct_val, label_val = struct_vals[1], label_vals[1]
        lines!(ax, [0, 1], [struct_val, label_val], linestyle=:solid)

        if metric != "Structural Distance"
            total_count += 1
            label_val > struct_val && (discordant_count += 1)
        end
    end

    if metric != "Structural Distance" && total_count > 0
        discordance_pct = 100 * discordant_count / total_count
        ax.title = @sprintf("%.2f%% discordant", discordance_pct)
    end

    occursin("MIST", metric) || ylims!(ax, (nothing, 1))
    xlims!(ax, (-0.5, 1.5))

    ax.spinewidth = 0.8
    ax.xgridvisible = false
    ax.ygridvisible = false
end

function plot_discordance(label_path="../../osmo_data/safe_to_share.csv",
                          triplets_path="../../osmo_data/triplets.csv",
                          anchor_path="../../osmo_data/anchor_labels.json")

    code_to_smiles = Dict(k => kekulize(v) for (k, v) in load_code_to_smiles(label_path, anchor_path))

    triplets = DataFrame(CSV.File(triplets_path))
    triplets[!, "Perceptual Distance"] ./= 100
    filter!(row -> row.roleA == "anchor", triplets)

    triplets_mean = combine(groupby(triplets, ["RedJade Code A", "RedJade Code B", "roleB"]),
                            names(triplets, Real) .=> mean .=> names(triplets, Real))

    mist_pred_full, mist_pred, gnn_pred = validate_triplet_predictions()
    triplets_mean = compute_distance_metrics(triplets_mean, code_to_smiles,
                                             mist_pred_full, mist_pred, gnn_pred)

    distances = ["Structural Distance", "Perceptual Distance", "Predicted Label Distance (GNN)",
                 "Cosine Distance (MIST)", "Euclidean Distance (MIST)", "Cosine Distance (MIST Full)"]

    triples = load_triplets(triplets)
    triples_df = build_triples_dataframe(triples, triplets_mean, distances)

    with_theme(MISTStyle.theme()) do
        fig = Figure(size=(178mm, 110mm))
        gl = GridLayout(fig[1, 1])

        axes = [Axis(gl[div(i-1, 3)+1, mod1(i, 3)]; xlabel="", ylabel=distances[i],
                    titlesize=6,
                    xticks=([0, 1], ["Dissimilar\nStructure", "Similar\nStructure"]))
                for i in 1:6]

        linkyaxes!(axes[1], axes[2], axes[3])

        for (ax, metric) in zip(axes, distances)
            plot_metric_panel!(ax, filter(r -> r.metric == metric, triples_df), metric)
        end

        for (i, label) in enumerate(["a", "b", "c", "d", "e", "f"])
            sublabel!(gl[div(i-1, 3)+1, mod1(i, 3), TopLeft()], label; left=15pt)
        end

        MISTStyle.savefig("discordant", fig)
    end
end
