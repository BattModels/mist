using Makie
using MISTStyle
using DataFrames
using JSON3
using Statistics
using Printf
using PythonCall: pyimport, pyconvert, pylist
using CSV: CSV
using Distances: pairwise, Euclidean, CosineDist
using LinearAlgebra: norm
using Colors: distinguishable_colors, RGB

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
    transformers, torch = pyimport("transformers"), pyimport("torch")
    model = transformers.AutoModel.from_pretrained(model_path; trust_remote_code=true)
    model.eval()
    device = pyconvert(Bool, torch.cuda.is_available()) ? torch.device("cuda") :
             pyconvert(Bool, torch.backends.mps.is_available()) ? torch.device("mps") : torch.device("cpu")
    model.to(device)
end

function load_code_to_smiles(label_path="../../osmo_data/safe_to_share.csv",
                              anchor_path="../../osmo_data/anchor_labels.json")
    labels = DataFrame(CSV.File(label_path))
    merge!(Dict(zip(labels[!, "Sample Identifier"], labels.SMILES)),
           JSON3.read(read(anchor_path, String), Dict{String,String}))
end

function get_olfactory_profiles(model, smiles_list::Vector{String}; batch_size=32, mc_iterations=10)
    kekule_smiles = kekulize.(smiles_list)
    model.train()
    mc_logits, channel_names = [], nothing

    for _ in 1:mc_iterations
        all_logits = []
        for i in 1:batch_size:length(kekule_smiles)
            output = model.predict(pylist(kekule_smiles[i:min(i+batch_size-1, end)]))
            isnothing(channel_names) && (channel_names = sort(collect(pyconvert(Vector{String}, output.keys()))))
            push!(all_logits, reduce(hcat, [pyconvert(Vector{Float64}, output[ch]["value"]) for ch in channel_names]))
        end
        push!(mc_logits, vcat(all_logits...))
    end

    model.eval()
    mean_logits = mean(mc_logits)
    (mean_logits, @.(1 / (1 + exp(-mean_logits))), channel_names)
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
    all_logits, _, channel_names = get_olfactory_profiles(model, df.smiles)

    mist_pred_full = DataFrame(all_logits, channel_names)
    mist_pred = mist_pred_full[:, names(df)[2:end-1]]
    kek_smiles = kekulize.(df.smiles)
    mist_pred_full.smiles = mist_pred.smiles = df.smiles = kek_smiles

    (mist_pred_full, mist_pred, df[:, 2:end])
end

project_to_poincare(x; eps=1e-5) = (n = norm(x); n < 1 - eps ? x : x * (1 - eps) / n)

function hyperbolic_distance(u, v; eps=1e-5)
    u_proj, v_proj = project_to_poincare(u; eps), project_to_poincare(v; eps)
    diff_sq = sum(abs2, u_proj .- v_proj)
    denom = (1 - sum(abs2, u_proj)) * (1 - sum(abs2, v_proj))
    acosh(1 + 2 * diff_sq / max(denom, eps))
end

function pairwise_hyperbolic(X::Matrix; dims=1)
    n = size(X, dims)
    D = zeros(n, n)
    for i in 1:n, j in i+1:n
        row_i, row_j = dims == 1 ? (X[i, :], X[j, :]) : (X[:, i], X[:, j])
        D[i, j] = D[j, i] = hyperbolic_distance(row_i, row_j)
    end
    D
end

function load_triplets(triplets)
    triples = Tuple{String,String,String}[]
    for anchor in unique(triplets[triplets.roleA .== "anchor", "RedJade Code A"])
        mask = (triplets.roleA .== "anchor") .& (triplets[!, "RedJade Code A"] .== anchor)
        label_cliffs = unique(triplets[mask .& (triplets.roleB .== "label_cliff"), "RedJade Code B"])
        struct_cliffs = unique(triplets[mask .& (triplets.roleB .== "struct_cliff"), "RedJade Code B"])
        for l in label_cliffs, s in struct_cliffs
            push!(triples, (anchor, s, l))
        end
    end
    @info "$(length(triples)) triples loaded"
    triples
end

function compute_distance_metrics(triplets_mean, code_to_smiles, mist_pred_full, mist_pred, gnn_pred)
    pred_dfs = (mist_pred_full, mist_pred, gnn_pred)
    indices = [Dict(zip(df.smiles, 1:nrow(df))) for df in pred_dfs]
    matrices = [Matrix(select(df, Not(:smiles))) for df in pred_dfs]

    dist_configs = [
        (pairwise(CosineDist(), matrices[2], dims=1), "Cosine Distance (MIST)", indices[2]),
        (pairwise(Euclidean(), matrices[2], dims=1), "Euclidean Distance (MIST)", indices[2]),
        # (pairwise_hyperbolic(matrices[2], dims=1), "Hyperbolic Distance (MIST)", indices[2]),
        # (pairwise(CosineDist(), matrices[1], dims=1), "Cosine Distance (MIST Full)", indices[1]),
        (pairwise(CosineDist(), matrices[3], dims=1), "Predicted Label Distance (GNN)", indices[3])
    ]

    for (dist_mat, name, idx_dict) in dist_configs
        triplets_mean[!, name] = [dist_mat[idx_dict[code_to_smiles[r["RedJade Code A"]]],
                                           idx_dict[code_to_smiles[r["RedJade Code B"]]]] for r in eachrow(triplets_mean)]
    end
    triplets_mean
end

function build_triples_dataframe(triples, triplets_mean, distances)
    lookup = Dict((r["RedJade Code A"], r["RedJade Code B"], r.roleB) => i for (i, r) in enumerate(eachrow(triplets_mean)))
    triples_data = NamedTuple[]
    for (anchor, struct_cliff, label_cliff) in triples, metric in distances,
        (code, cliff_type) in [(struct_cliff, "struct"), (label_cliff, "label")]
        idx = get(lookup, (anchor, code, "$(cliff_type)_cliff"), nothing)
        !isnothing(idx) && push!(triples_data, (anchor=anchor, struct_cliff=struct_cliff, label=label_cliff,
                                                 metric=metric, cliff=cliff_type, value=triplets_mean[idx, metric]))
    end
    DataFrame(triples_data)
end

function plot_metric_panel!(ax, metric_data, metric)
    struct_data, label_data = filter(r -> r.cliff == "struct", metric_data).value,
                               filter(r -> r.cliff == "label", metric_data).value

    violin!(ax, fill(0.0, length(struct_data)), struct_data;
                                      color=(MISTStyle.UM_COLORS.maize, 0.5), width=0.8, datalimits=extrema)
    violin!(ax, fill(1.0, length(label_data)), label_data;
                                     color=(MISTStyle.UM_COLORS.blue, 0.5), width=0.8, datalimits=extrema)

    unique_triples = unique(collect(zip(metric_data.anchor, metric_data.struct_cliff, metric_data.label)))
    colors = distinguishable_colors(length(unique_triples), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    discordant_count, total_count = 0, 0

    for (idx, (a, s, l)) in enumerate(unique_triples)
        triple_data = filter(r -> r.anchor == a && r.struct_cliff == s && r.label == l, metric_data)
        struct_vals, label_vals = filter(r -> r.cliff == "struct", triple_data).value,
                                   filter(r -> r.cliff == "label", triple_data).value

        struct_val, label_val = struct_vals[1], label_vals[1]
        lines!(ax, [0, 1], [struct_val, label_val]; color=colors[idx], linestyle=:solid)
        metric != "Structural Distance" && (total_count += 1; label_val > struct_val && (discordant_count += 1))
    end

    metric != "Structural Distance" && total_count > 0 && (ax.title = @sprintf("%.2f%% discordant", 100discordant_count/total_count))
    occursin("MIST", metric) || ylims!(ax, (nothing, 1))
    xlims!(ax, (-0.5, 1.5))
    ax.spinewidth, ax.xgridvisible, ax.ygridvisible = 0.8, false, false
end

function plot_struct_vs_perceptual(triples_df, triples, code_to_smiles; metric="Euclidean Distance (MIST)")
    unique_triads = unique([(t[1], t[2], t[3]) for t in triples])
    colors = distinguishable_colors(length(unique_triads), [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    markers = [:circle, :rect, :diamond, :utriangle, :dtriangle, :star5, :hexagon, :cross, :xcross]

    metric_max = maximum(filter(r -> r.metric == metric, triples_df).value)
    normalize = occursin("Euclidean", metric)
    norm_val(v) = normalize ? v / metric_max : v
    get_val(df, m) = filter(r -> r.metric == m, df).value[1]
    csv_data = NamedTuple[]

    with_theme(MISTStyle.theme()) do
        fig = Figure(size=(55mm, 50mm), padding=(4,4,4,4))
        ax = Axis(fig[1,1]; xlabel="Structural Distance",
                  ylabel=normalize ? "Normalized $metric" : metric, aspect=1,
                  limits = ( (0.3, 1),  (0.3, 1)))

        for (idx, (anchor, sc, lc)) in enumerate(unique_triads)
            color, marker = colors[idx], markers[mod1(idx, length(markers))]

            for (cliff_type, stroke) in [("struct", nothing), ("label", :black)]
                row = filter(r -> r.anchor == anchor && r.struct_cliff == sc &&
                                  r.label == lc && r.cliff == cliff_type, triples_df)

                x, y = get_val(row, "Structural Distance"), norm_val(get_val(row, metric))
                scatter!(ax, [x], [y]; color, marker, strokewidth=isnothing(stroke) ? 0 : 0.6,
                         strokecolor=something(stroke, :black))
                push!(csv_data, (triad=idx, anchor=anchor, anchor_smiles=code_to_smiles[anchor],
                                 struct_cliff=sc, struct_cliff_smiles=code_to_smiles[sc],
                                 label_cliff=lc, label_cliff_smiles=code_to_smiles[lc],
                                 pair_type=cliff_type, x=x, y=y))
            end

            sr = filter(r -> r.anchor == anchor && r.struct_cliff == sc && r.label == lc && r.cliff == "struct", triples_df)
            lr = filter(r -> r.anchor == anchor && r.struct_cliff == sc && r.label == lc && r.cliff == "label", triples_df)
            lines!(ax,
                [get_val(sr, "Structural Distance"), get_val(lr, "Structural Distance")],
                [norm_val(get_val(sr, metric)), norm_val(get_val(lr, metric))]; color=(color, 0.5), linewidth=1)
        end
        axislegend(ax, [MarkerElement(color=:gray, marker=:circle, markersize=3),
                        MarkerElement(color=:gray, marker=:circle, strokewidth=0.6, strokecolor=:black, markersize=3)],
                   [ "Perceptually Similar", "Structurally Similar"]; position=:rb, padding=(1,1,1,1))

        MISTStyle.savefig("struct_vs_$(lowercase(replace(metric, ' ' => '_', '(' => "", ')' => "")))", fig)
    end

    csv_df = DataFrame(csv_data)
    CSV.write("fig/struct_vs_$(lowercase(replace(metric, ' ' => '_', '(' => "", ')' => ""))).csv", csv_df)
    @info "Saved $(nrow(csv_df)) points"
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
    triplets_mean = compute_distance_metrics(triplets_mean, code_to_smiles, mist_pred_full, mist_pred, gnn_pred)

    distances = ["Structural Distance", "Perceptual Distance", "Predicted Label Distance (GNN)",
                 "Cosine Distance (MIST)", "Euclidean Distance (MIST)", ]

    triples = load_triplets(triplets)
    triples_df = build_triples_dataframe(triples, triplets_mean, distances)

    with_theme(MISTStyle.theme()) do
        fig = Figure(size=(150mm, 75mm))
        gl = GridLayout(fig[1, 1])
        axes = [
            Axis(
                gl[div(i-1,3)+1, mod1(i,3)];
                xlabel="",
                ylabel=distances[i],
                spinewidth=0.8,
                xgridvisible = false,
                ygridvisible = false,
                xticks=([0,1], ["Dissimilar\nStructure", "Similar\nStructure"])
                ) for i in 1:5]
        linkyaxes!(axes[1], axes[2], axes[3])

        for (ax, metric) in zip(axes, distances)
            plot_metric_panel!(ax, filter(r -> r.metric == metric, triples_df), metric)
        end
        for (i, label) in enumerate('a':'e')
            sublabel!(gl[div(i-1,3)+1, mod1(i,3), TopLeft()], string(label); left=15pt)
        end
        MISTStyle.savefig("discordant", fig)
    end

    plot_struct_vs_perceptual(triples_df, triples, code_to_smiles; metric="Perceptual Distance")
    plot_struct_vs_perceptual(triples_df, triples, code_to_smiles; metric="Euclidean Distance (MIST)")
end
