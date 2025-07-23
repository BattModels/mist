using PythonCall
using DataFrames
using Makie
using CSV: CSV
using MISTStyle
using Format: format
using StatsBase: mean
using LinearAlgebra: diag

using Mixtures: clean_target_name

function label_smi(smi::String)
    known = Dict(
        "O" => "Water",
        "CC#N" => "ACN",
        "CC(=O)OCCOC(=O)C" => "EDGA",
        "CN1CCN(C)C1=O" => "DMI",
        "CN(CCO)CCO" => "MDEA",
    )
    return get(known, smi, smi)
end



# ACN & EDGA
rho = [1.0462, 1.0874, 0.792, 0.9414, 1.047, 0.966, 1.0844, 1.1106, 1.087]
x1 = [0.5, 0.25, 1, 0.75, 0.5, 0.75, 0.25, 0, 0]
mw_acn = 41.05
mw_edga = 176.21
rho_acn = 0.786
rho_edga = 1.104 #0.5 * (1.1106 + 1.087)
linear_mix(x, a, b) = x * a + (1 - x) * b
rho_excess = @. rho - linear_mix(x1, rho_acn, rho_edga)
mw_ideal = linear_mix.(x1, mw_acn, mw_edga)
mv = @. mw_ideal / rho
mv_excess = @. mv - linear_mix(x1, mw_acn/rho_acn, mw_edga/rho_edga)

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
        dfs = Mixtures.evaluate_mixtures(model, mixtures)
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
        out = subset(df, :compounds => ByRow(x -> smi1 in x && smi2 in x), :temperature => ByRow(==(temp)))
        # transform!(out,
        #     ["molar_volume_excess", "molar_volume"] => ByRow(/) => "molar_volume_excess",
        #     ["molar_volume_excess_ref", "molar_volume_ref"] => ByRow(/) => "molar_volume_excess_ref",
        #     ["density_excess", "density"] => ByRow(/) => "density_excess",
        #     ["density_excess_ref", "density_ref"] => ByRow(/) => "density_excess_ref",
        #     # ["molar_enthalpy_excess" "molar_enthalpy"] => ByRow(/) => "molar_enthalpy_excess",
        #     # ["molar_enthalpy_excess_ref", "molar_enthalpy_ref"] => ByRow(/) => "molar_enthalpy_excess_ref",
        # )
        return out
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

function argextreme(x, y)
    x = x[(!ismissing).(y)]
    y = y[(!ismissing).(y)]
    if length(x) == 0
        return missing
    end
    if abs(maximum(y)) > abs(minimum(y))
        idx = argmax(y)
    else
        idx = argmin(y)
    end
    return x[idx]
end

function maximum_skew(df)
    df = transform(df, :compounds => ByRow(sort) => :compound_id)
    skew = combine(groupby(df, :compound_id)) do gdf
        out = Dict()
        for target in ["density", "molar_volume", "molar_enthalpy"]
            # Model
            x1 = first.(gdf.composition)
            y = gdf[!, "$(target)_excess"]
            out[target] = abs(0.5 - argextreme(x1, y))

            # Reference
            y = gdf[!, "$(target)_excess_ref"]
            out["$(target)_ref"] = abs(0.5 - argextreme(x1, y))
        end
        return NamedTuple(Symbol(k) => v for (k, v) in pairs(out))
    end
end

function plot_excess_skewness(df)
    f = Figure(; size=(1.7inch, 1.5inch))
    skew = maximum_skew(df)

    ax = Axis(f[1, 1];
        xlabel="Reference Excess Asymmetry",
        ylabel="Predicted Excess Asymmetry",
        xtickformat="{:.0%}",
        ytickformat="{:.0%}",
        aspect=DataAspect(),
        limits=((0, 0.5), (0, 0.5)),
    )

    ablines!(ax, 0, 1; color=:black, linestyle=:dash)
    points = Point2[]
    text = []
    for (label, target) in [("Density", "density"), ("Molar Volume", "molar_volume"), ("Molar Enthalpy", "molar_enthalpy")]
        x = skew[!, "$(target)_ref"]
        y = skew[!, target]
        valid = @. !ismissing(y) && !ismissing(x)
        scatter!(ax, x, y; label)
        for row in eachrow(skew[valid, :])
            if "O" in row.compound_id
                smi1, smi2 = row.compound_id
                name1 = label_smi(smi1)
                name2 = label_smi(smi2)
                push!(text, "$name1 & $name2")
                push!(points, Point2(row.density_ref, row.density))
            end
        end
    end

    axislegend(ax; position=:rt)
    return f
end
