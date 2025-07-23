using PythonCall
using DataFrames
using Makie
using CSV: CSV
using MISTStyle
using Format: format
using StatsBase: mean
using LinearAlgebra: diag

pyexcess = pyimport("excess")


mixtures = [
    pydict(; compounds=@py(["CC#N", "CCOCCOCCOC(=O)C"]), temperature=293.15),
    pydict(; compounds=@py(["CC#N", "CCOCCOCCOC(=O)C"]), temperature=300),
]

function evaluate_mixtures(model, mixtures)
    targets = clean_target_name.(pyconvert(Vector{String}, model.config.target_columns))
    rows = map(pyexcess.evaluate_mixtures(model, mixtures)) do row
        row = pyconvert(Dict{String, Union{Float64, String, Vector}}, row)
        out = Dict(
            "compounds" => row["compounds"],
            "composition" => row["composition"],
            "temperature" => row["temperature"],
        )

        for (idx, target) in enumerate(targets)
            out[target] = row["y"][idx]
            out["$(target)_excess"] = row["y_excess"][idx]
            out["$(target)_linear"] = row["y_linear"][idx]
        end

        return out
    end
    return DataFrame(rows)
end

function plot_acn_edga(models::Vector{String})
    f = Figure()
    ax_excess = Axis(f[1, 1];
        limits=((0, 1), nothing),
        xlabel="Percent ACN",
        ylabel="Excess Density",
        xtickformat="{:.0%}",
        ytickformat="{:.0%}",
        xlabelvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
    )
    ax_volume = Axis(f[2, 1];
        limits=((0, 1), nothing),
        xlabel="ACN Mole Fraction",
        ylabel="Excess Molar Volume",
        xtickformat="{:.0%}",
        ytickformat="{:.0%}",
    )
    scatter!(ax_excess, x1, rho_excess ./ rho; label="Experimental")
    scatter!(ax_volume, x1, mv_excess ./ mv; label="Experimental")
    for model_path in models
        name = splitpath(model_path)[end-2]
        model = pyexcess.load_excess_model(model_path)
        dfs = evaluate_mixtures(model, mixtures)
        subset!(dfs, :temperature => ByRow(!=(300)))
        lines!(ax_excess, first.(dfs.composition), dfs.density_excess ./ dfs.density; label=name)
        lines!(ax_volume, first.(dfs.composition), dfs.molar_volume_excess ./ dfs.molar_volume; label=name)
    end
    # axislegend(ax_excess)
    axislegend(ax_volume; orientation=:horizontal)
    return f
end




function load_reference(path)
    df = DataFrame(CSV.File(path))
end

function dataframe_inspector(df, colums)
    function mixture_inspector(plot, idx, pos)
        entries = map(colums) do col
            data = df[idx, col]
            if data isa Real
                return string(col) * ": " * format(df[idx, col])
            elseif data isa AbstractVector{<:Real}
                return string(col) * ": " * join(format.(data), ", ")
            else
                return string(col) * ": " * string(data)
            end
        end
        return join(entries, "\n")
    end
    return mixture_inspector
end

mae(x, y) = mean(@.(abs(x - y)))
rmse(x, y) = sqrt(mean(@.((x - y)^2)))

unzip(a) = map(x->getfield.(a, x), fieldnames(eltype(a)))
skipmissingpairs(x...) = Iterators.filter(x -> all(!ismissing, x), zip(x...)) |> collect |> unzip

function compound_stats(df)
    compounds = Set(Iterators.flatten(df.compounds))
    rows = []
    for c in compounds
        df_c = subset(df, :compounds => ByRow(x -> c in x))
        row = Dict{String, Any}(
            "compound" => c,
            "nobs" => nrow(df_c),
        )
        for (name, rmetric) in [("mae", mae), ("rmse", rmse)]
            metric = (x, y) -> rmetric(skipmissingpairs(x, y)...)
            row["density_$(name)"] = metric(df_c.density, df_c.density_ref)
            row["density_excess_$(name)"] = metric(df_c.density_excess, df_c.density_excess_ref)
            row["molar_volume_$(name)"] = metric(df_c.molar_volume, df_c.molar_volume_ref)
            row["molar_enthalpy_$(name)"] = metric(df_c.molar_enthalpy, df_c.molar_enthalpy_ref)
            row["molar_volume_excess_$(name)"] = metric(df_c.molar_volume_excess, df_c.molar_volume_excess_ref)
        end
        push!(rows, row)
    end
    return DataFrame(rows)
end





function parity_plots(df, model)
    targets = clean_target_name.(pyconvert(Vector{String}, model.config.target_columns))
    compounds = unique(Iterators.flatten(df.compounds))

    mixture_selection = Observable((first(compounds), first(compounds), 293.15))
    function di(df, idx_obs, args...)
        df_inspect = dataframe_inspector(df, args...)
        function label(plot, idx, pos)
            smi1 = df[idx, :compounds][1]
            smi2 = df[idx, :compounds][2]
            temp = df[idx, :temperature]
            mixture_selection[] = (smi1, smi2, temp)
            return df_inspect(plot, idx, pos)
        end
        return label
    end

    f = Figure()
    gl_parity = GridLayout(f[1, 1])
    gl = GridLayout(gl_parity[1, 1])
    cb = Colorbar(gl_parity[1, 2]; colormap=:managua, colorrange=extrema(df.temperature))
    temperature = df[!, "temperature"]
    inspector_columns = ["temperature", "compounds", "composition"]
    for (tdx, target) in enumerate(targets)
        ax_y = Axis(gl[1, tdx])
        ablines!(ax_y, 0, 1; color=:black, linestyle=:dash)
        scatter!(ax_y, df[!, "$(target)_ref"], df[!, target];
            color=temperature,
            inspectable=true,
            inspector_label = di(df, mixture_selection, [target, "$(target)_ref", inspector_columns...]),
            MISTStyle.cb_attrs(cb, Scatter)...
        )

        ax_e = Axis(gl[2, tdx]; xlabel=target)
        ablines!(ax_e, 0, 1; color=:black, linestyle=:dash)
        scatter!(ax_e, df[!, "$(target)_excess_ref"], df[!, "$(target)_excess"];
            color=temperature,
            inspectable=true,
            inspector_label = di(df, mixture_selection, [target, "$(target)_ref", inspector_columns...]),
            MISTStyle.cb_attrs(cb, Scatter)...
        )

        if tdx == 1
            ax_y.ylabel[] = "Total"
            ax_e.ylabel[] = "Excess"
        end
    end

    gl_predict = GridLayout(f[2, 1])
    df = transform(df, :composition => ByRow(first) => :x1)
    dfs = lift(mixture_selection) do (smi1, smi2, temp)
        subset(df, :compounds => ByRow(x -> smi1 in x && smi2 in x), :temperature => ByRow(==(temp)))
    end
    for (tdx, target) in enumerate(targets)
        ax = Axis(gl_predict[1, tdx]; limits=((0, 1), extrema(df[!, target])))
        h = scatter!(ax, lift_points(dfs, "x1", target); color=:red)
        h = scatter!(ax, lift_points(dfs, "x1", "$(target)_ref"); color=:blue)
        on(_ -> autolimits!(ax), mixture_selection)

        ax = Axis(gl_predict[2, tdx]; limits=((0, 1), nothing))
        scatter!(ax, lift_points(dfs, "x1", "$(target)_excess"); color=:red)
        scatter!(ax, lift_points(dfs, "x1", "$(target)_excess_ref"); color=:blue)
        on(_ -> autolimits!(ax), mixture_selection)
    end
    predict_label = lift(mixture_selection) do (smi1, smi2, temp)
        return "$smi1 vs. $smi2 at $(format("{:.2f}", temp)) K"
    end
    Label(gl_predict[0, :], predict_label)

    elems = [
        LineElement(; color=:red, label="Prediction"),
        LineElement(; color=:blue, label="Reference"),
    ]
    Legend(gl_predict[3, :], elems, MISTStyle.label.(elems);
        nbanks=2,
        tellheight=true,
        tellwidth=false,
    )
    rowsize!(gl_predict, 0, Auto(0.2))
    rowsize!(gl_predict, 3, Auto(0.2))

    DataInspector(f)
    return f
end

function lift_points(df, x, y)
    lift(df) do df
        df = dropmissing(df[!, [x, y]])
        points = Point2f.(df[!, x], df[!, y])
        return points
    end
end

function plot_mixture(df, model)
    targets = clean_target_name.(pyconvert(Vector{String}, model.config.target_columns))

    f = Figure()
    gl = GridLayout(f[1, 1])
    ui = GridLayout(f[2, 1])
    compounds = unique(Iterators.flatten(df.compounds))
    smi1 = Menu(ui[1, 1], options=compounds)

    smi2_options = lift(smi1.selection) do smi1
        dfs = subset(df, :compounds => ByRow(x -> smi1 in x))
        unique(Iterators.flatten(dfs.compounds))
    end
    smi2 = Menu(ui[1, 2], options=smi2_options)

    df = transform(df, :composition => ByRow(first) => :x1)

    dfs = lift(smi1.selection, smi2.selection) do smi1, smi2
        subset(df, :compounds => ByRow(x -> smi1 in x && smi2 in x))
    end

    for (tdx, target) in enumerate(targets)
        ax = Axis(gl[1, tdx]; limits=((0, 1), nothing))
        scatter!(ax, lift_points(dfs, "x1", target); label="Prediction")
        scatter!(ax, lift_points(dfs, "x1", "$(target)_ref"); label="Reference")
        on(_ -> autolimits!(ax), smi2.selection)

        ax = Axis(gl[2, tdx]; limits=((0, 1), nothing))
        scatter!(ax, lift_points(dfs, "x1", "$(target)_excess"))
        scatter!(ax, lift_points(dfs, "x1", "$(target)_excess_ref"))
        on(_ -> autolimits!(ax), smi2.selection)
    end

    notify(smi1.selection)
    notify(smi2.selection)


    return f
end
