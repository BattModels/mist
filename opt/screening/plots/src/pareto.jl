function get_pareto_front(x::Vector, y::Vector; quad=:lt, ax=nothing)
    # quad options:
    # :lt = minimize x, maximize y (lower-left to top-right)
    # :ll = minimize x, minimize y (lower-left)
    # :rt = maximize x, minimize y (right-top to bottom-left)
    if quad == :lt
        # Minimize x, maximize y
        idx = Metaheuristics.get_non_dominated_solutions_perm(map(vcat, x, -1 .* y))
    elseif quad == :ll
        # Minimize both x and y
        idx = Metaheuristics.get_non_dominated_solutions_perm(map(vcat, x, y))
    elseif quad == :rt
        # Maximize x, minimize y
        idx = Metaheuristics.get_non_dominated_solutions_perm(map(vcat, -1 .* x, y))
    else
        error("quad=$quad not implemented")
    end
    front = Point2.(x[idx], y[idx])
    frontier_base = Point2.(sort(front; by=first, rev=true))
    if ax !== nothing
        limits = lift(ax.finallimits) do hr
            lx, ly = hr.origin
            ux, uy = hr.origin .+ hr.widths
            lx, ux = extrema([lx, ux])
            ly, uy = extrema([ly, uy])
            return (; lx, ly, ux, uy)
        end
        frontier = lift(limits) do limits
            if quad == :lt
                fs = Point2(limits.ux, frontier_base[1][2])
                fe = Point2(frontier_base[end][1], limits.ly)
                vcat([fs], frontier_base, [fe])
            elseif quad == :ll
                fs = Point2(limits.ux, frontier_base[1][2])
                fe = Point2(frontier_base[end][1], limits.uy)
                vcat([fs], frontier_base, [fe])
            elseif quad == :rt
                fs = Point2(frontier_base[1][1], limits.uy)
                fe = Point2(limits.lx, frontier_base[end][2])
                vcat([fs], frontier_base, [fe])
            else
                error("quad=$quad not implemented for axis extension")
            end
        end
        return frontier
    else
        return frontier_base
    end
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

function figure_screening(trace, case, ref, df_speed; dft=nothing)
    f = Figure(;
        size=(3.42inch, 2.7inch),
        figure_padding=(2, 2, 2, 5)
    )
    gl_perf = GridLayout(f[1, 1])
    plot_pareto_front!(GridLayout(f[2, 1]), case, ref; dft)
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

label_electrolyte_pareto(df) = label_electrolyte_pareto!(deepcopy(df))
function label_electrolyte_pareto!(case)
    bp = -case[!, :bp]
    gap = -case[!, :gap] .* HARTREE_TO_EV
    homo = case[!, :homo] .* HARTREE_TO_EV
    mp = case[!, :mp]
    canidates = map(vcat, homo, gap, mp, bp)
    front = Metaheuristics.get_non_dominated_solutions(canidates)
    nidx = findall(in(front), canidates)
    case.dominated .= true
    case.dominated[nidx] .= false
    sort!(case, :dominated; rev=true)
    return case
end

function plot_pareto_front(case, ref; dft=nothing)
    f = Figure(; size=(210pt, 100pt), figure_padding=(1, 3, 1, 2))
    plot_pareto_front!(f, case, ref; dft)
end
function plot_pareto_front!(f, case, ref; dft=nothing)
    mp_limits = extrema(vcat(case.mp, [0]))
    bp_limits = extrema(vcat(case.bp, [75]))
    pareto_kwargs = (;
        linewidth=1.5pt,
        alpha=0.7,
    )

    # Net non-dominated
    case = label_electrolyte_pareto(case)
    n_front = count(.!case.dominated)
    @info "non-dominated" sort(case[.!case.dominated, :], :inchi_key)
    @info "non-dominated" n_front nrow(case) n_front / nrow(case)

    # DFT (qmist) calculations, split into the generated and the reference molecules
    has_ref_dft = false
    if !isnothing(dft)
        dft_gen = subset(dft, :inchi_key => ByRow(∈(Set(case.inchi_key))))
        dft_homo = float.(dft_gen.homo) .* HARTREE_TO_EV
        dft_gap = float.(dft_gen.gap) .* HARTREE_TO_EV
        @info "DFT (qmist), generated" nrow(dft_gen) nrow(case) nrow(dft_gen) / nrow(case)

        # Only the reference molecules the generator rediscovered were ever sent to DFT,
        # so this front covers a subset of `ref` -- see the coverage logged below.
        if "inchi_key" in names(ref)
            dft_ref = subset(dft, :inchi_key => ByRow(∈(Set(ref.inchi_key))))
            has_ref_dft = nrow(dft_ref) > 1
            ref_dft_homo = float.(dft_ref.homo) .* HARTREE_TO_EV
            ref_dft_gap = float.(dft_ref.gap) .* HARTREE_TO_EV
            @info "DFT (qmist), reference" nrow(dft_ref) nrow(ref) nrow(dft_ref) / nrow(ref)
        end
    end

    ax = Axis(f[1, 1];
        limits=(mp_limits, bp_limits),
        xlabel=L"Melt ($\degree C$)",
        ylabel=L"Boil ($\degree C$)",
    )
    scatter_samples!(ax, case.mp, case.bp, case.dominated)
    stairs!(ax, get_pareto_front(ref.mp, ref.bp; ax);
        color=MISTStyle.UM_COLORS.maize,
        linestyle=:solid,
        pareto_kwargs...
    )
    stairs!(ax, get_pareto_front(case.mp, case.bp; ax);
        color=MISTStyle.UM_COLORS.blue,
        linestyle=:solid,
        pareto_kwargs...
    )

    ax = Axis(f[1, 2];
        # Widen to fit the DFT cloud, which runs up/right of the MIST predictions
        limits=isnothing(dft) ? ((-10.5, -7), (5, 13)) : ((-10.5, -5.5), (3.5, 13)),
        xlabel=L"HOMO (eV)$$",
        ylabel=L"Gap (eV)$$",
        xticks=WilkinsonTicks(5; k_max=7),
        yticks=WilkinsonTicks(5; k_max=7),
    )
    h, h_front = scatter_samples!(ax, case.homo .* HARTREE_TO_EV, case.gap .* HARTREE_TO_EV, case.dominated)
    h.label = "Generated"
    h_front.label = "On Pareto Front, Generated"
    if !isnothing(dft)
        h_dft = scatter!(ax, dft_homo, dft_gap;
            marker=:utriangle,
            color=MISTStyle.UM_COLORS.orange,
            alpha=0.35,
            markersize=3pt,
            label="Generated, DFT",
        )
    end
    # Colour encodes provenance (maize = reference, blue = generated); linestyle encodes the
    # level of theory (solid = MIST prediction, dotted = DFT/qmist).
    h_ref_front = lines!(ax, get_pareto_front(ref.homo .* HARTREE_TO_EV, ref.gap .* HARTREE_TO_EV; ax);
        color=MISTStyle.UM_COLORS.maize,
        linestyle=:solid,
        label="Ref. Pareto Front",
        pareto_kwargs...
    )
    if has_ref_dft
        h_ref_dft_front = lines!(ax, get_pareto_front(ref_dft_homo, ref_dft_gap; ax);
            color=MISTStyle.UM_COLORS.maize,
            linestyle=:dot,
            label="Ref. Pareto Front, DFT",
            pareto_kwargs...
        )
    end
    h_gen_front = stairs!(ax, get_pareto_front(case.homo .* HARTREE_TO_EV, case.gap .* HARTREE_TO_EV; ax);
        color=MISTStyle.UM_COLORS.blue,
        linestyle=:solid,
        label="Generated Pareto Front",
        pareto_kwargs...
    )
    if !isnothing(dft)
        h_dft_front = stairs!(ax, get_pareto_front(dft_homo, dft_gap; ax);
            color=MISTStyle.UM_COLORS.blue,
            linestyle=:dot,
            label="Generated Pareto Front, DFT",
            pareto_kwargs...
        )
    end
    # Stairs! leaves linestyle unset...
    elems = [h, h_front,
        LineElement(; label=h_ref_front.label, linestyle=:solid, color=h_ref_front.color, linewidth=h_ref_front.linewidth),
        LineElement(; label=h_gen_front.label, linestyle=:solid, color=h_gen_front.color, linewidth=h_gen_front.linewidth),
    ]
    if has_ref_dft
        push!(elems, LineElement(; label=h_ref_dft_front.label, linestyle=:dot,
            color=h_ref_dft_front.color, linewidth=h_ref_dft_front.linewidth))
    end
    if !isnothing(dft)
        push!(elems,
            MarkerElement(; label=h_dft.label, marker=:utriangle, color=MISTStyle.UM_COLORS.orange),
            LineElement(; label=h_dft_front.label, linestyle=:dot, color=h_dft_front.color, linewidth=h_dft_front.linewidth),
        )
    end
    Legend(f[2, :], elems, MISTStyle.label.(elems);
        tellheight=true, tellwidth=true,
        orientation=:horizontal,
        nbanks=isnothing(dft) ? 1 : 3,
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

function save_pareto_front(df_mol)
    df_front = label_electrolyte_pareto(df_mol)
    subset!(df_front, :dominated => ByRow(!))
    transform!(df_front, [:bp, :mp] => ByRow(-) => :thermal_window)
    to_hartree = ByRow(x -> x*HARTREE_TO_EV)
    transform!(df_front,
        :homo => to_hartree => :homo,
        :lumo => to_hartree => :lumo,
        :gap => to_hartree => :gap,
    )
    sort!(df_front, [:gap, :thermal_window]; rev=true)

    df_front.id = 1:nrow(df_front)
    columns = [
        "id" => "#",
        "homo" => "HOMO",
        "lumo" => "LUMO",
        "gap" => "Gap",
        "mp" => LatexCell("Melt"),
        "bp" => LatexCell("Boil"),
    ]

    function fmt_lr(v, i, j)
        col = first(columns[j])
        if col in ["homo", "lumo", "gap"]
            return format("{:.1f}", v)
        elseif col in ["mp", "bp"]
            return format("{:.0f}", v)
        else
            return v
        end
    end

    # Latex Table
    table = pretty_table(String, df_front[!, first.(columns)];
        backend=:latex,
        table_format=LatexTableFormat(;
            # Rules only under the column labels, matching the old `hlines=[:header]`
            horizontal_line_at_beginning=false,
            horizontal_line_after_column_labels=true,
            horizontal_lines_at_data_rows=:none,
            horizontal_line_after_data_rows=false,
        ),
        formatters=[fmt_lr],
        alignment=[:c for idx in eachindex(columns)],
        column_labels=last.(columns),
    )
    # PrettyTables v3 dropped `table_type=:longtable`, so promote the tabular by hand -- the
    # front runs to dozens of rows and will not fit on a single page. `\endhead` repeats the
    # column labels after each page break.
    table = replace(table,
        "\\begin{tabular}" => "\\begin{longtable}",
        "\\end{tabular}" => "\\end{longtable}",
    )
    table = replace(table, "\\hline\n" => "\\hline\n  \\endhead\n"; count=1)
    return table, df_front
end
