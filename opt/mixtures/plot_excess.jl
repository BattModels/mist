using PythonCall
using DataFrames
using Makie
using CSV: CSV
using MISTStyle
using Format: format
using StatsBase: mean, cor, corspearman, weights, tiedrank
using LinearAlgebra: diag

using Mixtures: Mixtures, clean_target_name

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


function title!(s)
    words = split(lowercase(s))
    return join([uppercasefirst(word) for word in words], " ")
end



function plot_experimental_data!(f, model)
    
    ax_density = Axis(f[1, 1];
        limits=((0, 1), nothing),
        xlabel="ACN Mole Fraction",
        ylabel=L"$\rho^E$ (g/cm$^3$)",
        xlabelvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
    )
    ax_volume = Axis(f[2, 1];
        limits=((0, 1), nothing),
        xlabel="ACN Mole Fraction",
        ylabel=L"$V^E_m$ (cm$^3$/mol)",
    )
    mixtures = [
        pydict(; compounds=@py(["CC#N", "CC(=O)OCCOC(=O)C"]), temperature=299.25),
        pydict(; compounds=@py(["CC#N", "CCC1COC(=O)O1"]), temperature=299.25),
    ]
    dfs = Mixtures.evaluate_mixtures(model, mixtures)
    dfs.x1 = first.(dfs.composition)
    for gdf in groupby(dfs, ["compounds"])
        lines!(ax_volume, gdf.x1, gdf.molar_volume_excess; linewidth=1pt, linestyle=:solid
        )
        lines!(ax_density, gdf.x1, gdf.density_excess; linewidth=1pt, linestyle=:solid
        )
    end

    exp_dfs = [
        DataFrame(CSV.File("panel_data/experimental_ACN_EGDA.csv")) => "EGDA",
        DataFrame(CSV.File("panel_data/experimental_ACN_BC.csv")) => "BC"
    ]
    for (df, label) in exp_dfs
        scatter!(ax_volume, df.x1, df[!, "Excess Molar Volume [cm3/mol]"]; label=label)
        scatter!(ax_density, df.x1, df[!, "Excess Density [g/cm3]"]; label=label)
    end
    scatter!(ax_density, [-1, -1], [0, 0.13]; label="Experimental", color=:black)
    lines!(ax_density, [-1, -1], [0, 0.13]; label="MIST", color=:black)
    axislegend(ax_density; nbanks=2, padding=(1, 1, 1, 1), margin=(1, 1, 1, 1))
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

plot_excess_skewness(df) = plot_excess_skewness!(Figure(), df)
function plot_excess_skewness!(f, df)
    skew = Mixtures.excess_skew(df)

    ax = Axis(f[1, 1];
        xlabel="Reference Excess Asymmetry",
        ylabel="Predicted Excess Asymmetry",
        xtickformat="{:.0%}",
        ytickformat="{:.0%}",
        limits=((0, 0.5), (0, 0.5)),
        xticks=WilkinsonTicks(5),
        xminorticks=IntervalsBetween(5),
        xminorticksvisible=true,
        yticks=WilkinsonTicks(5),
        yminorticks=IntervalsBetween(5),
        yminorticksvisible=true,
        xscale=sqrt,
        yscale=sqrt,
    )

    cb = Colorbar(f[1, 2];
        label="Max. Abs. Relative Excess",
        colormap=Reverse(:oslo),
        colorrange=(0, 0.1),
        tickformat="{:.0%}",
        tellheight=true,
        minorticksvisible=true,
        minorticks=IntervalsBetween(2),
    )
    targets = [
        ("Molar Volume", "molar_volume", :circle),
        ("Density", "density", :rect),
        ("Molar Enthalpy", "molar_enthalpy", :utriangle)
    ]
    for (label, target, marker) in targets
        df_target = select(skew, "compound_id", target, "$(target)_ref", "$(target)_rel_ref")
        dropmissing!(df_target)
        transform!(df_target, "$(target)_rel_ref" => ByRow(abs) => "$(target)_rel_ref")
        sort!(df_target, "$(target)_rel_ref"; rev=false)
        x = df_target[!, "$(target)_ref"]
        y = df_target[!, target]
        color = df_target[!, "$(target)_rel_ref"]
        nrow(df_target) == 0 && continue
        scatter!(ax, x, y;
            label,
            marker,
            color,
            strokewidth=0.25pt,
            strokecolor=MISTStyle.UM_COLORS.ash,
            MISTStyle.cb_attrs(cb, Scatter)...
        )

        # Compute and report correlations
        weight = weights(color)
        pearson = cor(x, y)
        pearson_weighted = cor(hcat(x, y), weight)[1, end]
        spearman = corspearman(x, y)
        spearman_weighted = cor(hcat(tiedrank(x), tiedrank(y)), weight)[1, end]
        @info "Excess Skew Correlations: $target" pearson spearman pearson_weighted spearman_weighted
    end

    axislegend(ax; position=:rt, theme(:Legend)...)
    return f
end

function plot_soap!(f)
    x_min = 0.3
    x_max = 1.1
    y_min = 0.0
    y_max = 0.035
    sim_threshold = 0.6
    excess_threshold = 0.025
    ax1 = Axis(f[1, 1];
        limits=((x_min, 1), (0, y_max)),
        xlabel="Similarity",
        ylabel=L"Maximum Absolute $V^{E}_m$",
        yticks=WilkinsonTicks(3; k_min = 3, k_max=5)
    )
    df = DataFrame(CSV.File("panel_data/soap_similarity.csv"))

    df[!, :abs_target] = abs.(
        df[!,:"excess molar volume [centimeter ** 3 / mole]"] ./
        df[!,:"molar volume [centimeter ** 3 / mole]"]
    )
    transform!(df, ["smi1", "smi2" ] => ByRow((x, y) -> sort([x, y])) => :compound_id)
    df = combine(groupby(df, [:compound_id])) do gdf
        idx = argmax(gdf.abs_target)
        return (; 
            max_abs_relative_vol = gdf.abs_target[idx], 
            name1 = title!(first(gdf.name1)), 
            name2 = title!(first(gdf.name2)), 
            similarity = first(gdf.similarity),
            temperature = gdf[idx, "temperature [kelvin]"]
        )
    end

    dropmissing!(df)
    transform!(df, ["name1", "name2" ] => ByRow((x, y) -> "$x\n$y") => :labels)

    scatter!(
        ax1, df[!, "similarity"],  df[!, "max_abs_relative_vol"];
        color = (MISTStyle.UM_COLORS.blue, 0.5), marker=:circle,
        strokewidth=0.25pt, strokecolor=:black
    )

    df_up = subset(df,
            :similarity => ByRow(>(sim_threshold)),
            :max_abs_relative_vol => ByRow(>(excess_threshold)),
        )
    CSV.write("panel_data/df_up.csv", df_up)

    points = Point2.(df_up.similarity, df_up.max_abs_relative_vol)
    annotation!(ax1, points; text =  df_up.labels, fontsize=5pt, shrink = (2.0, 2.0))

    df_left = subset(df,
        :similarity => ByRow(<(sim_threshold)),
        :max_abs_relative_vol => ByRow(<(excess_threshold)),
    )
    CSV.write("panel_data/df_left.csv", df_left)

    points = Point2.(df_left.similarity, df_left.max_abs_relative_vol)
    annotation!(ax1, points; text =  df_left.labels, fontsize=5pt, shrink = (2.0, 2.0))
    return f
end

function plot_soap_outliers!(f, model)
    ax2 = Axis(f[1, 1];
        limits=((0, 1), (-2.4, nothing)),
        xlabel=L"$x_1$",
        ylabel=L"$V^{E}_m \;$ (cm$^3$/mol)",
    )
    mixtures = [
        pydict(; compounds=@py(["CC1COC(=O)O1", "ClC(Cl)Cl"]), temperature=298.15),
        pydict(; compounds=@py(["NCCN", "OCCO"]), temperature=298.15),
        pydict(; compounds=@py(["CN(CCO)CCO", "O"]), temperature=298.15),
        pydict(; compounds=@py(["CN1CCN(C)C1=O", "O"]), temperature=298.15)
    ]
    df = Mixtures.evaluate_mixtures(model, mixtures; n=50, gradients=false)
    df.x1 = first.(df.composition)

    data = DataFrame(CSV.File("panel_data/soap_similarity.csv"))

    for gdf in groupby(df, ["compounds"])

        @info nrow(gdf)
        df_ref = subset(data,
            :smi1 => ByRow(in(gdf.compounds[1])),
            :smi2 => ByRow(in(gdf.compounds[1])),
            "temperature [kelvin]" => ByRow(==(298.15))
        )
        name1 = title!(df_ref.name1[1])
        name2 = title!(df_ref.name2[1])
        label = "$name1 & $name2"
        @info nrow(df_ref)
        lines!(ax2, gdf.x1, gdf.molar_volume_excess; label = label, linewidth=1pt, linestyle=:solid
        )

        if nrow(df_ref)> 2
            dropmissing!(df_ref, "excess molar volume [centimeter ** 3 / mole]")
            scatter!(ax2, df_ref[!, "x1"], df_ref[!, "excess molar volume [centimeter ** 3 / mole]"];
            )
        end
    end
    axislegend(ax2, position = :rb, padding=(1, 1, 1, 1), margin=(1, 1, 1, 1))
    return f
end

function plot_ionic_conductivity!(f)

    cb = Colorbar(f[2, 1:2];
        label = L"Temperature (K)$$",
        colormap = MISTStyle.CONTINUOUS_COLORS,
        colorrange = (240, 340),
        flipaxis = false,
        vertical=false,
        tellheight=true,
        tellwidth=true
    )

    df = DataFrame(CSV.File("panel_data/ionic_conductivity_curves.csv"))
    df = df[1:2:end, :]
    # List the two salts (panels) in order
    salts = unique(df.salt_name)
    line_styles = [:solid, :dot]
    # Sort temperatures and build a continuous, dark‐cropped colormap
    temps = sort(unique(df.temperature))

    ax = Axis(f[1, 1];
            xlabel = L"x_{Li}",  
            ylabel = L"\sigma (mS/cm)",
            limits= ((0, 0.20), (0, 0.35)),
            yticksvisible=true,
            yticklabelsvisible=true,
    )
    x_min = 0.160369437447523
    x_max = 0.20
    y_min = 0.0
    y_max = 0.35
    poly!(ax, Point2f[(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)], color = (MISTStyle.UM_COLORS.maize, 0.2), strokewidth = 0)
    # Plot each temperature curve in both panels
    for (idx, salt) in enumerate(salts), T in temps
        mask = (df.salt_name .== salt) .& (df.temperature .== T)
        sub = df[mask, :]
        label = (salt == "LiPF6") ? L"LiPF$_6$" : salt
        lines!(
            ax,
            sub.composition,
            sub.predictions;
            linewidth = 1pt,
            linestyle = line_styles[idx],
            color     = sub.temperature,
            label=label,
            MISTStyle.cb_attrs(cb, Lines)...
        )
    end
    axislegend(ax, position = :rt, padding=(1, 1, 1, 1), margin=(1, 1, 1, 1), unique=true)
    return f
end

function plot_ionic_temperature!(f)
    ax = Axis(f[1, 1];
        xlabel="Temperature (K)",
        ylabel=L"$\sigma$ (mS/cm)",
    )
    df = DataFrame(CSV.File("panel_data/temperature_w8phe03q.csv"))
    errorbars!(ax, df[!, "temperature"], df[!, "predicted"], df[!, "st_dev"], color= :black)
    scatter!(ax, df[!, "temperature"], df[!, "predicted"],  markersize = 3)
    data_T_max = 293.150000
    x_min, x_max = extrema(df[!, "temperature"])
    y_min, y_max = extrema(df[!, "predicted"])
    poly!(Point2f[(x_min, y_min), (x_min, y_max), (data_T_max, y_max), (data_T_max, y_min)], color = MISTStyle.UM_COLORS.maize, alpha = 0.2 , strokewidth = 0)

    return f
end

function mixture_panel(model::Py, df_excess::DataFrame)
    model = Mixtures.load_excess_model(model_path)
    f = Figure(figure_padding = 5; size=(89mm, 190mm))
    plot_excess_skewness!(f[1, 1], df_excess)
    plot_soap!(f[1, 2])
    plot_soap_outliers!(f[2, 2], model)
    # plot_ionic_temperature!(f[3, 1])
    plot_ionic_conductivity!(f[3, :])
    plot_experimental_data!(f[2, 1], model)

    sublabel!(f[1, 1, TopLeft()], "a"; left=15pt)
    sublabel!(f[1, 2, TopLeft()], "b"; left=15pt)
    sublabel!(f[2, 1, TopLeft()], "c"; left=15pt)
    sublabel!(f[2, 2, TopLeft()], "d"; left=15pt)

    return f
end


function mixture_panel(model_id::String, excess_dataset::String)
    root_dir = realpath(joinpath(pkgdir(Mixtures), "..", ".."))
    excess_dataset = joinpath(root_dir, "excess_v5.csv")
    model = Mixtures.load_excess_model(joinpath(root_dir, "models", model_id)).to("mps")
    df_excess = Mixtures.evaluate_binary_csv(model, excess_dataset)

    return mixture_panel(model, df_excess)
end
