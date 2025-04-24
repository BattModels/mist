using MISTStyle: UM_COLORS
using Makie
using MISTStyle
using Metaheuristics
using DataFrames
using SQLite: SQLite
using CSV: CSV
using JSON: JSON

const HARTREE_TO_EV = 27.211_386_245_981

function load_data(path)
    file = !endswith(path, ".sqlite") ? joinpath(path, "merged.sqlite") : path
    db = SQLite.DB(file)
    df = nothing
    try
        df = SQLite.DBInterface.execute(db, "SELECT * from molecules") |> DataFrame
    finally
        close(db)
    end
    cols = keys(JSON.parse(df[1, :props]))
    transform!(df, :props => ByRow(JSON.parse) => Symbol.(cols))
    select!(df, Not(:props))
    return df
end

function load_all()
    cases = Dict{String,DataFrame}()
    for case in readdir(joinpath(@__DIR__, "out"); join=true)
        isfile(joinpath(case, "merged.sqlite")) || continue
        cases[basename(case)] = load_data(case)
    end
    return cases
end

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
        color=map(d -> d ? MISTStyle.CAT_COLORS[1] : UM_COLORS.blue, case.dominated),
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
        color=map(d -> d ? MISTStyle.CAT_COLORS[1] : UM_COLORS.blue, case.dominated),
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

function gen_plots()
    dfs = load_all()
    df_ref = DataFrame(CSV.File(joinpath(@__DIR__, "..", "design", "electrolytes_predictons.csv")))

    # MP < 0degC, BP > 75 degC, HOMO < -7eV, GAP > 5eV
    case_names = Dict(
        "1 gpus" => "4e64e005-d988-48d4-8879-1f7a7d5f2be9",
        "4 gpus" => "dcd68d1a-8190-4145-b77c-6486ac1a880b",
        "16 gpus" => "09b342fa-95ad-4727-ac0f-fee0b8c4b84b",
        "32 gpus" => "597f3b8a-9390-41b2-adc6-a974afd04c71",
        "40 gpus" => "f1c03e9a-ee30-4671-873b-b4e2f9be17a1",
    )

    for (name, id) in pairs(case_names)
        df = dfs[id]
        @info "$name - $id" nrow(df) sum(df.duplicate_count)
    end

    with_theme(MISTStyle.theme()) do
        plot_generation(
            Dict(name => dfs[id] for (name, id) in case_names)
        ) |> MISTStyle.savefig("generation")
        for (name, id) in pairs(case_names)
            plot_pareto_front(dfs[id], df_ref) |> MISTStyle.savefig(joinpath("pareto", name * "-" * id))
        end
    end
end

function (@main)(::Any)
    gen_plots()
end
