function get_pareto_front(x::Vector, y::Vector; quad=:lt, ax=nothing)
    @assert quad == :lt "not implemented"
    idx = Metaheuristics.get_non_dominated_solutions_perm(map(vcat, x, -1 .* y))
    front = Point2.(x[idx], y[idx])
    frontier = Point2.(sort(front; by=first, rev=true))
    if ax !== nothing
        limits = lift(ax.finallimits) do hr
            lx, ly = hr.origin
            ux, uy = hr.origin .+ hr.widths
            lx, ux = extrema([lx, ux])
            ly, uy = extrema([ly, uy])
            return (; lx, ly, ux, uy)
        end
        frontier = lift(limits) do limits
            fs = Point2(limits.ux, frontier[1][2])
            fe = Point2(frontier[end][1], limits.ly)
            vcat([fs], frontier, [fe])
        end
    end
    return frontier
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

function plot_gen_trace(args...; kwargs...)
    f = Figure(; size=(2inch, 1inch))
    plot_gen_trace!(f, args...; kwargs...)
end

function fit_exp_decay(x, y)
    df = DataFrame(x=x, y=float.(y[end] .- y))
    subset!(df, :y => ByRow(>(0)))
    m = glm(@formula(y ~ x), df, Normal(), LogLink())
    A = exp.(coef(m)[1])
    τ = coef(m)[2]
    y_hat = @. A * (1 - exp(x * τ))
    return (; m, y_hat, A, τ)
end

weak_scaling(df_speed) = weak_scaling!(Figure(), df_speed)
function weak_scaling!(f, df_speed)
    ax = Axis(f[1, 1];
        limits=((nothing, 9), (0, nothing)),
        xlabel="GPUs",
        ylabel="Evaluated/GPU-sec",
        xscale=log2,
    )
    # x = df_speed.gpus .+ 0.1 * randn(nrow(df_speed))
    x = df_speed.gpus .* (1 .+ 0.02 .* randn(nrow(df_speed)))
    h = scatter!(ax,
        x, df_speed.global_throughput ./ df_speed.gpus;
        color=df_speed.batch_size,
        marker=:circle,
        colormap=:roma,
        alpha=0.8,
    )
    Colorbar(f[1, 2], h; label="Batch Size")
    return f
end

function figure_screening(trace, case, ref, df_speed)
    f = Figure(;
        size=(3.42inch, 2.5inch),
        figure_padding=(2, 2, 2, 5)
    )
    gl_perf = GridLayout(f[1, 1])
    plot_pareto_front!(GridLayout(f[2, 1]), case, ref)
    plot_gen_trace!(GridLayout(gl_perf[1, 1]), trace)

    gl = GridLayout(gl_perf[1, 2])
    weak_scaling!(gl, df_speed)
    colgap!(gl_perf, 4pt)

    sublabel!(gl_perf[1, 1, TopLeft()], "a"; left=27pt)
    sublabel!(gl_perf[1, 2, TopLeft()], "b"; left=25pt)
    sublabel!(f[2, 1][1, 1, TopLeft()], "c"; left=15pt)
    sublabel!(f[2, 1][1, 2, TopLeft()], "d"; left=5pt)

    resize_to_layout!(f)

    return f
end

function plot_gen_trace!(f, trace)
    ax = Axis(f[1, 1];
        xlabel="Wall Time [s]",
        ylabel="Evaluated",
        limits=((0, nothing), (0, 100e6)),
        xlabelvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
        yticks=WilkinsonTicks(3),
        yminorticks=IntervalsBetween(5),
        yminorticksvisible=true,
    )

    m_uniq = fit_exp_decay(trace.time, trace.unique_molecules)
    m_pass = fit_exp_decay(trace.time, trace.n_passing)
    uniq_max_init_rate = -m_uniq.A * m_uniq.τ
    pass_max_init_rate = -m_pass.A * m_pass.τ
    @info "Initial Rates" uniq_max_init_rate pass_max_init_rate m_uniq.m m_pass.m

    lines!(ax, trace.time, trace.unique_molecules)
    ax2 = Axis(f[2, 1];
        xlabel="Wall Time [s]",
        ylabel="Passing",
        limits=((0, nothing), (0, 2100)),
        xminorticksvisible=true,
        yminorticksvisible=true,
        xminorticks=IntervalsBetween(10),
        yminorticks=IntervalsBetween(5),
        yticks=[0, 1000, 2000],
    )
    lines!(ax2, trace.time, trace.n_passing)
    linkxaxes!(ax, ax2)

    return f
end

function plot_pareto_front(case, ref)
    f = Figure(; size=(210pt, 100pt), figure_padding=(1, 3, 1, 2))
    plot_pareto_front!(f, case, ref)
end
function plot_pareto_front!(f, case, ref)
    mp_limits = extrema(vcat(case.mp, [0]))
    bp_limits = extrema(vcat(case.bp, [75]))
    pareto_kwargs = (;
        linewidth=1.5pt,
        linestyle=:solid,
        alpha=0.7,
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
    @info "non-dominated" sort(case[nidx, :], :inchi_key)
    front = map(front) do p
        p[1] *= -1
        p[2] *= -1
        p
    end
    @info "non-dominated" length(front) nrow(case) length(front) / nrow(case)

    ax = Axis(f[1, 1];
        limits=(mp_limits, bp_limits),
        xlabel=L"Melt ($\degree C$)",
        ylabel=L"Boil ($\degree C$)",
    )
    scatter_samples!(ax, case.mp, case.bp, case.dominated)
    stairs!(ax, get_pareto_front(ref.mp, ref.bp; ax);
        color=MISTStyle.UM_COLORS.maize,
        pareto_kwargs...
    )
    stairs!(ax, get_pareto_front(case.mp, case.bp; ax);
        color=MISTStyle.UM_COLORS.blue,
        pareto_kwargs...
    )

    ax = Axis(f[1, 2];
        limits=((-10.5, -7), (5, 13)),
        xlabel=L"HOMO (eV)$$",
        ylabel=L"Gap (eV)$$",
        xticks=WilkinsonTicks(5; k_max=7),
        yticks=WilkinsonTicks(5; k_max=7),
    )
    h, h_front = scatter_samples!(ax, case.homo .* HARTREE_TO_EV, case.gap .* HARTREE_TO_EV, case.dominated)
    h.label = "Generated"
    h_front.label = "On Pareto Front, Generated"
    h_ref_front = lines!(ax, get_pareto_front(ref.homo .* HARTREE_TO_EV, ref.gap .* HARTREE_TO_EV; ax);
        color=MISTStyle.UM_COLORS.maize,
        label="Ref. Pareto Front",
        pareto_kwargs...
    )
    h_gen_front = stairs!(ax, get_pareto_front(case.homo .* HARTREE_TO_EV, case.gap .* HARTREE_TO_EV; ax);
        color=MISTStyle.UM_COLORS.blue,
        label="Generated Pareto Front",
        pareto_kwargs...
    )
    # Stairs! leaves linestyle unset...
    elems = [h, h_front,
        LineElement(; label=h_ref_front.label, linestyle=:solid, color=h_ref_front.color, linewidth=h_ref_front.linewidth),
        LineElement(; label=h_gen_front.label, linestyle=:solid, color=h_gen_front.color, linewidth=h_gen_front.linewidth),
    ]
    Legend(f[2, :], elems, MISTStyle.label.(elems);
        tellheight=true, tellwidth=true,
        orientation=:horizontal,
    )

    return f
end

function scatter_samples!(ax, x, y, dominated)
    h1 = scatter!(ax, x[dominated], y[dominated];
        marker=:circle,
        color=MISTStyle.CAT_COLORS[1],
        alpha=0.4,
    )
    h2 = scatter!(ax, x[.!dominated], y[.!dominated];
        marker=:star5,
        color=MISTStyle.UM_COLORS.blue,
    )
    return h1, h2
end
