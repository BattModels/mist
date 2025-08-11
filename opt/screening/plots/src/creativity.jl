function molecular_surprise(model::Py, smiles::Vector{String}; batch_size=32)
    _eval(batch) =  pyconvert(Vector{Float64}, model.score(PyList(batch)))
    batch_transcode(smiles) do batch
        pyconvert(Vector{Float64}, model.score(PyList(batch)))
    end
end

first_token(E::AbstractArray{<:Real, 3}) = eachrow(reshape(E[:, 1, :], size(E, 1), :))
first_token(E::AbstractMatrix) = vec(E[1, :])
mean_pooling(E::AbstractArray{<:Real, 3}) = eachrow(reshape(mean(E; dims=2), size(E, 1), :))
mean_pooling(E::AbstractMatrix) = vec(mean(E; dims=1))
no_pooling(E) = eachslice(E; dims=1)

function mist_embedding(
    model::Py,
    smiles::Vector{String};
    batch_size=32,
    pooling=first_token,
    collect=x -> stack(x; dims=1),
)
    batch_transcode(smiles; collect) do batch
        E = pyconvert(Array{Float64}, model.embed(PyList(batch)).to("cpu"))
        return pooling(E)
    end
end
function batch_transcode(f, smiles; batch_size=32, collect=Base.collect)
    transcode = __data_utils[].MolEncoding("smiles-kekule")
    smiles = map(x -> pyconvert(String, transcode(x)), smiles)
    ds = Iterators.partition(smiles, batch_size)
    out = Iterators.map(f, ds)
    return collect(Iterators.flatten(out))
end

function compare_creativity(
    run::String,
    chembl_dir::String,
    df_ref::DataFrame;
    mol_surprise::Py,
    mist_mp::Py,
    mist_bp::Py,
    kwargs...
)


    df_mol = load_generated_molecules(run)
    df_chembl = load_qmist_results(chembl_dir)
    df = vcat(
        _select_creativity(df_mol, mol_surprise, "Generated"),
        _select_creativity(df_chembl, mol_surprise, "ChEMBL"),
        _select_creativity(df_ref, mol_surprise, "Electrolytes"),
    )

    mp_missing = ismissing.(df.mp)
    df.mp[mp_missing] .= predict_mist(mist_mp, df.smiles[mp_missing]).mp
    bp_missing = ismissing.(df.bp)
    df.bp[bp_missing] .= predict_mist(mist_bp, df.smiles[bp_missing]).bp


    return compare_creativity(df), df
end

function compare_creativity(df::DataFrame, M = missing)
    f = Figure(; size=(3.42inch, 1inch))
    df = subset(df, :group => ByRow(in(["Generated", "Electrolytes"])))
    df.group = categorical(df.group; levels=["Generated", "Electrolytes"])
    df.gap .*= HARTREE_TO_EV
    df.homo .*= HARTREE_TO_EV
    ax1 = Axis(f[1, 1];
        ylabel="Molecular Surprise",
        xticks=(1:2, unique(df.group)),
        limits=((0, nothing), nothing),
        ygridvisible=true,
        yminorticksvisible=true,
        xticklabelrotation=0.3,
    )
    rainclouds!(ax1, levelcode.(df.group), df.surprise;
        gap=0.2,
        clouds=hist,
        jitter_width=0.1,
        boxplot_width=0.15,
        side_nudge=0.25,
        strokewidth=0.5,
        whiskerwidth=1.0,
        color=map(g -> MISTStyle.CAT_COLORS[levelcode(g)], df.group),
    )

    df = combine(groupby(df, :group)) do gdf
        pos = map(vcat, -gdf.gap, gdf.homo, gdf.mp, -gdf.bp)
        idx = Metaheuristics.get_non_dominated_solutions_perm(pos)
        h_pareto = mean(gdf.surprise[idx])
        h_std_pareto = std(gdf.surprise[idx])
        h_dominated = mean(gdf.surprise[Not(idx)])
        h_std_dominated = std(gdf.surprise[Not(idx)])
        h_delta = h_pareto - h_dominated
        h_std_delta = hypot(h_std_pareto, h_std_dominated)
        @info first(gdf.group) mean(gdf.surprise) std(gdf.surprise) mean(gdf.surprise[idx]) h_delta h_std_delta
        gdf.group_frontier .= false
        gdf[idx, :group_frontier] .= true
        return gdf
    end

    ax2 = Axis(f[1, 2];
        xlabel=L"$$HOMO [eV]",
        ylabel=L"$$Molecular Surprise",
        xticks=WilkinsonTicks(3),
    )
    sargs = (; markersize = 3)
    sort!(df, :group_frontier; rev=true)
    _mark_creative!(ax2, df, :homo, :surprise, df.group_frontier; sargs...)

    ax3 = Axis(f[1, 3];
        xlabel=L"$$Gap [eV]",
        ylabel=L"$$Molecular Surprise",
        xticks=WilkinsonTicks(3),
        yticks=ax1.yticks,
        ygridvisible=ax1.ygridvisible,
    )
    _mark_creative!(ax3, df, :gap, :surprise, df.group_frontier; sargs...)

    ax4 = Axis(f[1, 4];
        xlabel=L"$$Melt [$\degree C$ ]",
        xticks=WilkinsonTicks(3),
        ylabel=L"$$Molecular Surprise",
        yticks=ax1.yticks,
        ygridvisible=ax1.ygridvisible,
    )
    _mark_creative!(ax4, df, :mp, :surprise, df.group_frontier; sargs...)

    ax5 = Axis(f[1, 5];
        xlabel=L"$$Boil [$\degree C$ ]",
        xticks=WilkinsonTicks(3),
        ylabel=L"$$Molecular Surprise",
        yticks=ax1.yticks,
        ygridvisible=ax1.ygridvisible,
    )
    _mark_creative!(ax5, df, :bp, :surprise, df.group_frontier; sargs...)

    hideydecorations!(ax2; grid=false)
    hideydecorations!(ax3; grid=false)
    hideydecorations!(ax4; grid=false)
    hideydecorations!(ax5; grid=false)
    linkyaxes!(ax1, ax2, ax3)

    return f
end

function _annotate_creative!(ax, smiles, df, x, y; label_pos=nothing, kwargs...)
    key = inchi_key(smiles)
    @show row = first(df[df.inchi_key .== key, :])
    if label_pos !== nothing
        lx, ly = (row[x], row[y]) .+ label_pos
        return annotation!(ax, label_pos..., row[x], row[y]; kwargs...)
    else
        return annotation!(ax, row[x], row[y]; kwargs...)
    end
end

function _mark_creative!(ax, df, x, y, pareto; kwargs...)
    idx = findall(pareto)
    df_other = df[Not(idx), :]
    scatter!(ax, df_other[!, x], df_other[!, y];
        marker=:circle,
        color=MISTStyle.CAT_COLORS[levelcode.(df_other.group)],
        alpha=0.1,
        kwargs...
    )
    df_pareto = df[idx, :]
    scatter!(ax, df_pareto[!, x], df_pareto[!, y];
        marker=:star5,
        strokewidth=0.1,
        color=MISTStyle.CAT_COLORS[levelcode.(df_pareto.group)],
        kwargs...
    )
end

function _select_creativity(df::DataFrame, model::Py, group::String)
    df = _select_creativity(df, model)
    df.group .= group
    return df
end
function _select_creativity(df::DataFrame, model::Py)
    if "smi" in names(df)
        df = rename(df, "smi" => "smiles")
    end
    cols = ["inchi_key", "smiles", "homo", "gap", "mp", "bp"]
    df = select(df, intersect(cols, names(df)))
    df.surprise = molecular_surprise(model, df.smiles)
    df.embed = eachrow(mist_embedding(model, df.smiles))
    for col in cols
        if !(col in names(df))
            df[!, col] .= missing
        end
    end
    return df
end

bounding_diameter(embeddings::Matrix) = norm(map(x -> -(extrema(x)...), eachcol(e)))
function embedding_momemt(embeddings::Matrix, n=1)
    center = vec(mean(embeddings; dims=1))
    @assert size(center) == (size(embeddings, 2),) == (512,)
    return mean(eachrow(embeddings)) do emb
        return norm(emb .- center)^n
    end
end
function embedding_spread(embeddings::Matrix)
    center = vec(mean(embeddings; dims=1))
    return map(e -> norm(e .- center), eachrow(embeddings))
end

eculidean_distance(a, b) = norm(a .- b)
cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))
cosine_distance(a, b) = sqrt(2 * max(1 - cosine_similarity(a, b), 0))
angular_distance(a, b) = acos(min(max(cosine_similarity(a, b), 0), 1))

function embedding_mst_distance(embeddings::Matrix; distance=eculidean_distance)
    n = size(embeddings, 1)
    dd = Matrix{}(undef, n, n)
    for I in eachindex(IndexCartesian(), dd)
        x = embeddings[I[1], :]
        y = embeddings[I[2], :]
        @assert length(x) == length(y)
        dd[I] = distance(x, y)
    end
    g = complete_graph(n)
    o = boruvka_mst(g, dd)
    return o.weight
end

function renyi_entropy_estimate(embeddings; kwargs...)
    E = stack(embeddings; dims=1)
    @assert E isa AbstractMatrix
    mst = embedding_mst_distance(E; kwargs...)
    n, d = size(E)
    γ = (d - 1) / d
    renyi = (1 / (1-γ)) * (log(mst) - γ * log(n))
    return (; renyi, mst)
end
