function evaluate_odor(model::Py, smiles::Vector{String}; batch_size=32, filter_non_active=true)
    df = predict_mist(model, smiles; batch_size)

    # Summarize odor activations
    odor_summary = describe(df[!, Not("smiles")], :max, Base.Fix1(count, >(0)) => :nactive)
    filter_non_active && subset!(odor_summary, :max => ByRow(>(0)))
    sort!(odor_summary, :max; rev=true)

    df.max_odor = map(eachrow(df)) do row
        maximum(Vector(row[Not("smiles")]))
    end
    select!(df, "smiles", "max_odor", odor_summary.variable...)
    sort!(df, :max_odor; rev=true)

    @info "Top Odors" odor_summary
    return df
end

""" Return the names of the channels in the MIST model"""
channel_names(model::Py) = pyconvert(Vector{String}, [chn["name"] for chn in model.channels])

filter_non_active(df::DataFrame, odor_model::Py; kwargs...) = filter_non_active(df, channel_names(odor_model); kwargs...)

function filter_non_active(df::DataFrame, odor_columns=Not("smiles"); limit=nothing)
    odor_columns = intersect(odor_columns, names(df))
    odor_summary = describe(df[!, odor_columns], :max, Base.Fix1(count, >(0)) => :nactive)
    non_odor = names(df[!, Not(odor_columns)])
    subset!(odor_summary, :max => ByRow(>(0)))
    sort!(odor_summary, :max; rev=true)
    if !isnothing(limit)
        @show odor_summary = odor_summary[1:limit, :]
    end
    return select(df, non_odor..., odor_summary.variable...)
end

function count_active(df, odor_model::Py)
    scents = pyconvert(Vector{String}, [chn["name"] for chn in odor_model.channels])
    n = Dict()
    for scent in scents
        n[scent] = count(df[!, scent] .> 0)
    end
    s = collect(keys(n))
    counts = collect(values(n))
    return DataFrame(scent=s, active_count=counts)
end

logistic(x) = inv(1 + exp(-x))

function correlation_matrix(df::DataFrame, columns; correlation=cor)
    D = Matrix{Float64}(undef, length(columns), length(columns))
    for I in eachindex(IndexCartesian(), D)
        if I[1] == I[2]
            D[I] = 1
        elseif I[1] > I[2]
            D[I] = correlation(df[!, columns[I.I[1]]], df[!, columns[I.I[2]]])
            D[I[2], I[1]] = D[I]
        end
    end
    return D, columns
end

function figure_hclust(dist::Matrix, labels::Vector; kwargs...)
    c = hclust(dist; linkage=:single, branchorder=:barjoseph)
    dist = dist[c.order, c.order]
    labels = labels[c.order]

    f = Figure()
    cb = Colorbar(f[1, 2];
        colorrange=(-1, 1),
        colormap=:vik10,
        tellheight=true,
        tellwidth=true,
        halign=:left,
        valign=:top,
    )
    ticks = (eachindex(labels), labels)
    ax = Axis(f[1, 1];
        xticks=ticks, yticks=ticks,
        xticklabelrotation=0.55,
        xticklabelsvisible=false,
        xticksvisible=false,
        aspect=DataAspect(),
    )
    heatmap!(ax, dist; MISTStyle.cb_attrs(cb, Heatmap)..., kwargs...)
    return f
end

function linear_correlation(df, cols, target; correlation=cor)
    S = Matrix(df[!, cols])
    c = S \ df[!, target]
    P = sortperm(abs.(c))
    rho = Vector{Float64}(undef, length(cols))
    for i in range(1, length(cols))
        Sp = @view S[:, P[1:i]]
        c = Sp \ df[!, target]
        rho[i] = correlation(Sp * c, df[!, target])
    end
    return rho
end

function higher_order_odors(df)
    groups = Dict(
        "fruity & Sweet" => ["fruity", "tropical", "apple", "banana", "pear", "pineapple", "sweet"],
        "Floral" => ["floral", "rose", "green", "herbal", "ethereal"],
        "Pungent" => ["sulfurous", "onion", "garlic", "cheesy", "phenolic", "vegetable"],
    )
    out = Dict()
    for (label, odors) in pairs(groups)
        out[label] = vec(maximum(Matrix(df[!, odors]); dims=2))
    end
    dfo = DataFrame(out)
    dfo.smiles = df.smiles
    return dfo
end

function pick_odor(df, scent)
    df = deepcopy(df)
    df.group = categorical(df.group)
    dfi = subset(df, scent => ByRow(>(0)))
    dfo = subset(df, scent => ByRow(<(0)))
    f = Figure(; size=(105, 187))
    ax = Axis(f[1, 1]; xlabel="Homo", ylabel="Gap", limits=((-0.4, -0.25), nothing))
    kwargs = (; colormap=:tab10, colorrange=(1, 10), marker=:circle)
    scatter!(ax, dfo.homo, dfo.gap; color=levelcode.(dfo.group), alpha = 0.1, markersize=3pt, kwargs...)
    scatter!(ax, dfi.homo, dfi.gap; color=levelcode.(dfi.group), kwargs...)
    ax = Axis(f[2, 1]; xlabel="Melt", ylabel="Boil", limits=((-100, 0), (0, nothing)))
    scatter!(ax, dfo.mp, dfo.bp; color=levelcode.(dfo.group), alpha = 0.1, markersize=3pt, kwargs...)
    scatter!(ax, dfi.mp, dfi.bp; color=levelcode.(dfi.group), kwargs...)
    return f
end


function plot_odor_act(df)
    f = Figure(; size=(8inch, 8inch))
    odors = setdiff(names(df), ["smiles", "name", "class", "max_odor"])
    mols = map((n,c) -> "$n ($c)", df.name, df.class)
    ax = Axis(f[1, 1];
        xticks=(1:length(odors), odors),
        yticks=(1:length(mols), mols),
        xticklabelrotation=pi/4,
    )
    act = logistic.(Matrix(df[!, odors]))
    heatmap!(ax, act'; colormap=:lipari)
    return f
end

function plot_odor_tsne(df::DataFrame, odor_columns=Not("smiles"))
    act = Matrix(df[!, odor_columns])
    M = fit(ManifoldLearning.LLE, act')
    f = plot_odor_tsne(M, df, act, odor_columns)
    return f, M
end
plot_odor_tsne(M, df::DataFrame, model::Py; kwargs...) = plot_odor_tsne(M, df, channel_names(model); kwargs...)
function plot_odor_tsne(M, df::DataFrame,  act, odor_columns=Not("smiles"))
    act = Matrix(df[!, odor_columns])
    R = predict(M)
    f = Figure(; size=(2inch, 2inch))
    ax = Axis(f[1, 1])
    hidedecorations!(ax)
    dominate = map(argmax, eachrow(act))
    h = scatter!(ax, eachrow(R)...;
        marker=:circle,
        color=dominate,
        markersize=2pt,
        colormap=:tab10,
        colorrange=(1, 10),

    )

    # elements = map(enumerate(odor_columns)) do (i, label)
    #     PolyElement(; color=i, label, colormap=h.colormap, colorrange=h.colorrange)
    # end
    # Legend(f[1, 1], elements, MISTStyle.label.(elements);
    #     tellheight=false, tellwidth=false,
    #     # orientation=:horizontal,
    #     halign=:left,
    #     valign=:top,
    #     nbanks=2,
    # )
    # ann = annotation!(ax, Point2.(eachcol(R)); text=df.name)
    return f
end

function figure_odor_tsne(df::DataFrame, odor_model::Py; limit=10)
    scents = pyconvert(Vector{String}, [chn["name"] for chn in odor_model.channels])
    df_odor = filter_non_active(df, scents; limit)
    active_scents = intersect(names(df_odor), scents)
    df_odor.dominate = map(eachrow(df_odor)) do row
        return argmax(row[active_scents])
    end
    df_odor.embed = eachrow(mist_embedding(odor_model, df_odor.smiles; pooling=no_pooling))
    return df_odor
end

function figure_odor_tsne(df::DataFrame)
    E = stack(df.embed; dims=2)
    M = fit(ManifoldLearning.Isomap, E)
    # M = fit(ManifoldLearning.TSNE, E)
    return M
end

function figure_odor_tsne(df::DataFrame, M)
    f = Figure(; size=(2inch, 2inch))
    R = predict(M)
    ax = Axis(f[1, 1])
    df.dominate = categorical(string.(df.dominate))
    df = df[randperm(nrow(df)), :]
    scatter!(ax, eachrow(R)...;
        marker=:circle,
        markersize=2pt,
        color=levelcode.(df.dominate),
        colormap=:tab10,
        colorrange=(1, 10),
    )
    return f
end

function odor_counts(df::DataFrame, odor_model::Py)
    df_sum = combine(groupby(df, :group)) do gdf
        nactive = ScreeningPlots.count_active(gdf, odor_model)
        nactive.active_count ./= nrow(gdf)
        return nactive
    end
    @info df_sum
    groups = unique(df_sum.group)
    df_sum = unstack(df_sum, :group, :active_count)
    transform!(df_sum, groups => ByRow((x...) -> sum(x)) => :any_active)
    subset!(df_sum, :any_active => ByRow(>(0)))
    sort!(df_sum, :any_active; rev=true)
    scents = df_sum.scent
    @info df_sum

    df_sum  = stack(df_sum, groups; variable_name="group")
    df_sum.group = categorical(df_sum.group; levels=groups)
    df_sum.scent = categorical(df_sum.scent; levels=scents)
    return df_sum
end

function figure_odor_counts(df::DataFrame)
    f = Figure(; size=(2inch, 3.25inch))
    figure_odor_counts!(f, df)
    return resize_to_layout!(f)
end

function figure_odor_counts!(f, df::DataFrame)
    scents = levels(df.scent)
    ax = Axis(f[1, 1];
        limits=((2e-4, 1), nothing),
        yticks=(1:length(scents), scents),
        xlabel="Molecules with Odor",
        xscale=log10,
        xticks=([1e-3, 1e-2, 1e-1], ["0.1%", "1%", "10%"]),
        xticklabelsize=7pt,
        yticklabelsize=7pt,
    )
    h = barplot!(ax, levelcode.(df.scent), df.value;
        color=levelcode.(df.group),
        dodge=levelcode.(df.group),
        colormap=MISTStyle.CAT_COLORS,
        colorrange=(1, length(MISTStyle.CAT_COLORS)),
        direction=:x,
        fillto=1e-5,
    )

    elements = map(enumerate(unique(df.group))) do (i, label)
        PolyElement(; color=i, label, colormap=h.colormap, colorrange=h.colorrange)
    end
    Legend(f[1, 1], elements, MISTStyle.label.(elements);
        orientation=:horizontal,
        tellheight=false,
        tellwidth=false,
        halign=:right,
        valign=:top,
    )
    return f
end

function plot_pareto_front_scent(f, df, scent)
    f = Figure(; size=(100, 203))
    f = plot_pareto_front_scent!(f, df, scent)
    resize_to_layout!(f)
    return f
end
function plot_pareto_front_scent!(f, df, scent)
    df = subset(df, :group => ByRow(in(["Generated", "Electrolytes"])))
    df.group = categorical(df.group; levels=["Generated", "Electrolytes"])
    df = combine(first, groupby(df, [:inchi_key, :group]))
    pareto_kwargs = (;
        linewidth=1.5pt,
        alpha=0.7,
    )

    # Get non-dominated solutions
    df = combine(groupby(df, :group)) do gdf
        pos = map(vcat, -gdf.gap, gdf.homo, gdf.mp, -gdf.bp)
        idx = Metaheuristics.get_non_dominated_solutions_perm(pos)
        h_pareto = mean(gdf.surprise[idx])
        h_std_pareto = std(gdf.surprise[idx])
        h_dominated = mean(gdf.surprise[Not(idx)])
        h_std_dominated = std(gdf.surprise[Not(idx)])
        h_delta = h_pareto - h_dominated
        h_std_delta = hypot(h_std_pareto, h_std_dominated)
        gdf.group_frontier .= false
        gdf[idx, :group_frontier] .= true
        return gdf
    end


    # Convert Units
    df.gap .*= HARTREE_TO_EV
    df.homo .*= HARTREE_TO_EV

    skwargs = (;
        marker=:circle,
        colormap=MISTStyle.CAT_COLORS,
        colorrange=(1, length(MISTStyle.CAT_COLORS)),
    )

    ax = Axis(f[1, 1];
        limits=((-10.5, -7), (5, 13)),
        xlabel=L"HOMO (eV)$$",
        ylabel=L"Gap (eV)$$",
        xticks=WilkinsonTicks(5; k_max=7),
        yticks=WilkinsonTicks(5; k_max=7),
    )
    gen = subset(df, :group => ByRow(==("Generated")))
    w_scent = subset(df, scent => ByRow(>(0)))
    stairs!(ax, get_pareto_front(gen.homo, gen.gap; ax);
        color=MISTStyle.UM_COLORS.blue,
        pareto_kwargs...
    )
    scatter!(ax, df.homo, df.gap; color=levelcode.(df.group), alpha=0.2, markersize=3pt, skwargs...)
    scatter!(ax, w_scent.homo, w_scent.gap;
        color=levelcode.(w_scent.group),
        markersize=4pt,
        strokewidth=0.5,
        skwargs...
    )

    ax = Axis(f[2, 1];
        limits=((minimum(df.mp), 0), (50, maximum(df.bp))),
        xlabel=L"Melt ($\degree C$)",
        ylabel=L"Boil ($\degree C$)",
    )
    h_pareto = stairs!(ax, get_pareto_front(gen.mp, gen.bp; ax);
        color=MISTStyle.UM_COLORS.blue,
        pareto_kwargs...
    )
    w_scent = subset(df, scent => ByRow(>(0)))
    h_other = scatter!(ax, df.mp, df.bp; color=levelcode.(df.group), alpha=0.2, markersize=3pt, skwargs...)
    h = scatter!(ax, w_scent.mp, w_scent.bp;
        color=levelcode.(w_scent.group),
        markersize=4pt,
        strokewidth=0.5,
        skwargs...
    )

    elements = map(enumerate(unique(df.group))) do (i, label)
        PolyElement(; color=i, label, colormap=h.colormap, colorrange=h.colorrange)
    end |> Vector{Any}
    push!(elements, LineElement(;
        color=h_pareto.color,
        linestyle=h_pareto.linestyle,
        linewidth=h_pareto.linewidth,
        label="Pareto Front",
    ))
    push!(elements, MarkerElement(;
        label="Odorless",
        color=:gray,
        marker=h.marker,
        markersize=h.markersize,
        strokewidth=h.strokewidth,
    ))
    # push!(elements, MarkerElement(;
    #     label="Other Scents",
    #     color=:gray,
    #     marker=h_other.marker,
    #     markersize=h_other.markersize,
    #     strokewidth=h_other.strokewidth,
    # ))
    Legend(f[2, 1], elements, MISTStyle.label.(elements);
        tellheight=false, tellwidth=false,
        halign=:left,
        valign=:top,
    )

    return f
end
