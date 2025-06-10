function get_pareto_front(df, x, y; quad=:lt)
    @assert quad == :lt "not implemented"
    x = df[:, x]
    y = -df[:, y] # flip sign to maximize
    front = Metaheuristics.get_non_dominated_solutions(map(vcat, x, y))
    front = map(front) do p
        p[2] *= -1
        p
    end
    return Point2.(sort(front; by=first, rev=true))
end

function plot_generation(cases)
    f = Figure()
    ax = Axis(f[1, 1];
        xlabel="Duplicate Count",
        ylabel="Probability Density",
        limits=((0, nothing), (0, nothing)),
    )
    for (label, df) in pairs(cases)
        density!(ax, df.duplicate_count; label, bandwidth=0.75)
    end
    axislegend(ax)
    return f
end

function plot_pareto_front(case, ref)
    mp_limits = extrema(vcat(case.mp, [0]))
    bp_limits = extrema(vcat(case.bp, [75]))
    pareto_kwargs = (;
        marker=:star5,
        markersize=8pt,
    )

    # Net non-dominated
    bp = -case[!, :bp]
    gap = -case[!, :gap] .* HARTREE_TO_EV
    homo = case[!, :homo] .* HARTREE_TO_EV
    mp = case[!, :mp]
    canidates = map(vcat, homo, gap, mp, bp)
    front = Metaheuristics.get_non_dominated_solutions(canidates)
    nidx = findall(in(front), canidates)
    case = deepcopy(case)
    case.dominated .= true
    case.dominated[nidx] .= false
    sort!(case, :dominated; rev=true)
    @info "non-dominated" sort(case[nidx, :], :inchi)
    front = map(front) do p
        p[1] *= -1
        p[2] *= -1
        p
    end
    @info "non-dominated" length(front) nrow(case) length(front) / nrow(case)

    f = Figure(; size=(2inch, 1inch), figure_padding=(1, 3, 1, 2))
    ax = Axis(f[1, 1];
        limits=(mp_limits, bp_limits),
        xlabel=L"Melting Point [$\degree C$]",
        ylabel=L"Boiling Point [$\degree C$]",
    )
    scatter!(ax, case.mp, case.bp;
        marker=map(d -> d ? :circle : :star5, case.dominated),
        color=map(d -> d ? MISTStyle.CAT_COLORS[1] : MISTStyle.UM_COLORS.blue, case.dominated),
    )
    scatterlines!(ax, get_pareto_front(ref, :mp, :bp);
        color=MISTStyle.UM_COLORS.maize,
        pareto_kwargs...
    )
    scatterlines!(ax, get_pareto_front(case, :mp, :bp);
        color=MISTStyle.UM_COLORS.blue,
        pareto_kwargs...
    )

    ax = Axis(f[1, 2];
        limits=((-10, -7), (5, 12)),
        xlabel=L"HOMO [eV]$$",
        ylabel=L"Gap [eV]$$",
    )
    scatter!(ax, case.homo .* HARTREE_TO_EV, case.gap .* HARTREE_TO_EV;
        marker=map(d -> d ? :circle : :star5, case.dominated),
        color=map(d -> d ? MISTStyle.CAT_COLORS[1] : MISTStyle.UM_COLORS.blue, case.dominated),
        label="Generated",
    )
    scatterlines!(ax, get_pareto_front(ref, :homo, :gap) .* HARTREE_TO_EV;
        color=MISTStyle.UM_COLORS.maize,
        label="Ref. Pareto Front",
        pareto_kwargs...
    )
    scatterlines!(ax, get_pareto_front(case, :homo, :gap) .* HARTREE_TO_EV;
        color=MISTStyle.UM_COLORS.blue,
        label="Generated Pareto Front",
        pareto_kwargs...
    )
    Legend(f[2, :], ax;
        tellheight=true, tellwidth=false,
        orientation=:horizontal,
    )

    return f
end
