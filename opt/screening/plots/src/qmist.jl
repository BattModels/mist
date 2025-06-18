function load_qmist_results(folder::String)
    rows = []
    for file in readdir(folder; join=true)
        endswith(file, ".json") || continue
        push!(rows, JSON.parsefile(file))
    end
    df = DataFrame(rows)
    rename!(df, "InChIKey" => "inchi_key")
    return df
end


function predict_mist(model::Py, smi::Vector{String}; batch_size=32)
    model = model.eval()
    transcode = __data_utils[].MolEncoding("smiles-kekule")
    smi = map(x -> pyconvert(String, transcode(x)), smi)
    ds = Iterators.partition(smi, batch_size)
    channels = pyconvert(Vector{String}, [chn["name"] for chn in model.channels])
    out = Iterators.map(ds) do batch
        y = model.predict(PyList(batch); return_dict=false)
        yj = pyconvert(Matrix{Float64}, y)
        return eachrow(yj)
    end
    data = stack(collect(Iterators.flatten(out)))'
    df = DataFrame(data, channels)
    df.smiles = smi
    return df
end

function load_jsonl(file::String)
    rows = []
    open(file, "r") do io
        for line in eachline(io)
            push!(rows, JSON.parse(line))
        end
    end
    return DataFrame(rows)
end

function merge_qmist_results(qmist::DataFrame, ref::DataFrame)
    cols = filter(!=("smiles"), names(ref))
    df = innerjoin(qmist, ref; on=:smiles, renamecols = "_qmist" => "_qm9")
    return df, cols
end

figure_parity(args...; kwargs...) = figure_parity!(Figure(; size=(3.42inch, 2inch)), args...; kwargs...)
function figure_parity!(f, df::DataFrame, cols::Vector{String}; ref="_qm9", other="_qmist", label::Union{Pair{String,String},Nothing}=nothing)
    nrow = floor(Int, sqrt(length(cols)))
    ncol = ceil(Int, length(cols) / nrow)

    gl = GridLayout(f[1, 1])
    for (idx, col) in enumerate(cols)
        i, j = divrem(idx - 1, ncol)
        i += 1
        x = df[!, col * ref]
        y = df[!, col * other]
        xlim = extrema(x)
        ylim = extrema(y)
        ax = Axis(gl[i, j];
            title=format("{}\nρ: {:.2f}, MAE: {:.3f}", col, cor(x,y), mae(x .- y)),
            limits=MISTStyle.parity_limits(x, y),
            xticks=WilkinsonTicks(2),
            yticks=WilkinsonTicks(2),
        )
        if "color$(ref)" in names(df)
            sargs = (; alpha= 0.3, color=df[!, "color$(ref)"], colormap=MISTStyle.CAT_COLORS, colorrange=(1, 10))
        else
            sargs = (; alpha=0.3)
        end
        scatter!(ax, x, y; marker=:circle, alpha=0.3, sargs...)
        ablines!(ax, 0, 1; color=:black, linestyle=:dash)
    end

    if !isnothing(label)
        xlabel, ylabel = label
        Label(f[:, 0], ylabel, rotation=pi/2, tellwidth=true, tellheight=false)
        Label(f[end+1,:], xlabel,; tellwidth=false, tellheight=true)
    end

    if "color$(ref)" in names(df)
        @info "Group Sizes" combine(groupby(df, "color$(ref)"), DataFrames.nrow => :n)
        ng = count(==(1), df[!, "color$(ref)"])
        ni = count(==(2), df[!, "color$(ref)"])
        elements = map(enumerate(["Generated (n=$ng)", "Inventory (n=$ni)"])) do (idx, label)
            MarkerElement(; label, marker=:circle, color=MISTStyle.CAT_COLORS[idx])
        end
        Legend(gl[begin, end], elements, MISTStyle.label.(elements);
            fontsize=6pt,
            margin=(2pt, 2pt, 2pt, 2pt),
            padding=2pt,
            valign=:bottom,
            halign=:right,
        )
    end

    resize_to_layout!(f)
    return f
end

# Compare MIST vs. qmist
function compare_qmist(
    run::String,
    qmist_dir::String,
    cols::Vector{String} = ["gap", "lumo", "homo"];
    kwargs...
)
    # Load Screening & QMist results
    df_qmist = load_qmist_results(qmist_dir)
    df_mist = load_generated_molecules(run)
    select!(df_mist, ["inchi_key", cols...])
    select!(df_qmist, ["inchi_key", cols...])
    return compare_qmist(df_qmist, df_mist, cols; on="inchi_key", kwargs...)
end

# Compare MIST vs. qmist
function compare_qmist(
    run::String,
    qmist_dir::String,
    chembl_dir::String,
    mist_qm9::Py;
    cols::Vector{String} = ["gap", "lumo", "homo"],
    kwargs...
)
    # Load Screening & QMist results
    df_qmist = load_qmist_results(qmist_dir)
    df_mist = load_generated_molecules(run)
    select!(df_mist, ["inchi_key", cols...])
    select!(df_qmist, ["inchi_key", cols...])
    df_qmist.color .= 1
    df_mist.color .= 1

    # Load Chembl results
    df_chembl = load_qmist_results(chembl_dir)
    df_mist_chembl = predict_mist(mist_qm9, df_chembl.smiles)
    df_mist_chembl.inchi_key = df_chembl.inchi_key
    select!(df_mist_chembl, ["inchi_key", cols...])
    select!(df_chembl, ["inchi_key", cols...])
    df_chembl.color .= 2
    df_mist_chembl.color .= 2

    df_qmist = vcat(df_qmist, df_chembl)
    df_mist = vcat(df_mist, df_mist_chembl)

    return compare_qmist(df_qmist, df_mist, cols; on="inchi_key", kwargs...)
end

function compare_qmist(df_qmist::DataFrame, df_mist::DataFrame, cols::Vector{String}; on="InChIKey" => "inchi", kwargs...)
    df = innerjoin(df_qmist, df_mist; on, renamecols="_qmist" => "_mist")
    disallowmissing!(df)

    f = Figure(; size=(3.42inch, 1inch))
    figure_parity!(f ,df, cols; ref="_qmist", other="_mist", kwargs...)
    return f
end

function hit_rate!(df::DataFrame, limits)
    df_cols = names(df)
    select!(df, filter(!endswith("_hit"), names(df)))
    chns = String[]
    models = String[]
    for (col, limit) in limits
        for dc in filter(x -> occursin(col, x), df_cols)
            push!(chns, col)
            push!(models, split(dc, "_")[2])
            df[!, Symbol(dc * "_hit")] = map(x -> inbounds(x, limit), df[!, dc])
        end
    end
    chns = unique(chns)
    models = unique(models)

    # Mark overall hit rate
    for model in models
        transform!(df,
            map(col -> "$(col)_$(model)_hit", chns) => ByRow((x...) -> all(x)) => "$(model)_hit"
        )
    end

    return chns, models
end

function retrival_quality(df::DataFrame, chns, models::Pair{String})
    hitcols = filter(endswith("_hit"), names(df))
    df_hit = select(df, hitcols)
    ref_model, other_model = models
    cf = Dict{String, NamedTuple}()
    for chn in chns
        x = df_hit[!, "$(chn)_$(ref_model)_hit"]
        x_hat = df_hit[!, "$(chn)_$(other_model)_hit"]
        cf[chn] = confmatrix(x_hat, x)
    end
    cf["all"] = confmatrix(df_hit[!, other_model * "_hit"], df_hit[!, ref_model * "_hit"])
    return retrival_quality(cf)
end

retrival_quality(cfs::Dict) = Dict(k => retrival_quality(v) for (k, v) in cfs)

function retrival_quality(cf::NamedTuple)
    return (;
        cf,
        precision = cf.tp / (cf.tp + cf.fp),
    )
end

function confmatrix(x_hat::Vector{Bool}, x::Vector{Bool})
    return (;
        tp = sum(x_hat .& x),
        tn = sum(.!x_hat .& .!x),
        fp = sum(x_hat .& .!x),
        fn = sum(.!x_hat .& x),
    )
end

inbounds(x::Real, lim::Tuple) = inbounds(x, first(lim), last(lim))
inbounds(x::Real, lb::Real, ub::Real) = x >= lb && x <= ub
inbounds(x::Real, lb::Nothing, ub::Real) = x <= ub
inbounds(x::Real, lb::Real, ub::Nothing) = x >= lb
inbounds(x::Real, lb::Nothing, ub::Nothing) = true

