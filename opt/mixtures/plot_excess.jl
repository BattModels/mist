using PythonCall
using DataFrames
using Makie
using CSV: CSV
using MISTStyle
using Format: format
using JSON
using StatsBase: mean, cor, corspearman, weights, tiedrank
using JSON: JSON

using Mixtures: Mixtures, clean_target_name

# Directory containing the data release files
const DATA_DIR = realpath(joinpath(pkgdir(Mixtures), "..", "..", "data"))

function label_smi(smi::AbstractString)
    known = Dict(
        "O" => "Water",
        "CC#N" => "ACN", # Acetonitrile
        "CC(=O)OCCOC(=O)C" => "EDGA", # Ethylene Glycol Diacetate
        "CN1CCN(C)C1=O" => "DMI", # 1,3-Dimethyl-2-imidazolidinone
        "CN(CCO)CCO" => "MDEA", # N-Methyldiethanolamine
        "NCCN" => "1,2-DE", # 1,2-Diaminoethane
        "CC1COC(=O)O1" => "PC", # Propylene carbonate
        "OCCO" => "1,2-ED", # 1,2-Ethanediol
        "C(CO)O" => "1,2-ED", # 1,2-Ethanediol
        "ClC(Cl)Cl" => "Chloroform",
    )
    return get(known, smi, smi)
end

function titlecase(s)
    words = split(lowercase(s))
    return join([uppercasefirst(word) for word in words], " ")
end

function plot_experimental_data!(f, model; xaxisvisible=true)
    ax = Axis(f[1, 1];
        limits=((0, 1), (nothing, 0.15)),
        xlabel=L"x_1",
        ylabel=L"Relative $\rho^E$",
        xtickformat="{:.0%}",
        ytickformat="{:.0%}",
        xticksvisible=xaxisvisible,
        xlabelvisible=xaxisvisible,
        xticklabelsvisible=xaxisvisible,
    )

    mixtures = [
        Dict( "compounds" => ["CC#N", "CC(=O)OCCOC(=O)C"], "temperature" => 299.25),
        Dict( "compounds" => ["CC#N", "CCC1COC(=O)O1"], "temperature" => 299.25),
    ]
    dfs = Mixtures.evaluate_mixtures(model, mixtures; gradients=false)
    dfs.x1 = first.(dfs.composition)
    for gdf in groupby(dfs, ["compounds"])
        lines!(ax, gdf.x1, gdf.density_excess./gdf.density; linestyle=:solid)
    end

    exp_dfs = [
        DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "experimental_ACN_EGDA.csv"))) => "ACN & EGDA",
        DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "experimental_ACN_BC.csv"))) => "ACN & BC"
    ]

    elems = PolyElement[]
    for (df, label) in exp_dfs
        h = scatter!(ax, df.x1, df[!, "Excess Density [g/cm3]"]./df[!, "Density (g/cm^3)"] ; label)
        push!(elems, PolyElement(; color=h.color, label=h.label))
    end

    Legend(f[1, 1], elems, MISTStyle.label.(elems);
        halign=:left,
        valign=:top,
        orientation=:horizontal,
    )

    return f, ax
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
        xlabel="Reference Asymmetry",
        ylabel="Predicted Asymmetry",
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
        label="Max Abs. Rel. Excess",
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

    margin = something(theme(:Legend), (; margin=(1, 1, 1, 1))).margin
    axislegend(ax; position=:rb, margin)
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
        xlabel="SOAP Similarity",
        ylabel=L"Max $\left| V^{E}_m \right|$",
        xtickformat="{:.0%}",
        yticks=WilkinsonTicks(3; k_min = 3, k_max=5)
    )
    df = DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "soap_similarity.csv")))

    df[!, :abs_target] = abs.(
        df[!,:"excess molar volume [centimeter ** 3 / mole]"] ./
        df[!,:"molar volume [centimeter ** 3 / mole]"]
    )
    transform!(df, ["smi1", "smi2" ] => ByRow((x, y) -> sort([x, y])) => :compound_id)
    df = combine(groupby(df, [:compound_id])) do gdf
        idx = argmax(gdf.abs_target)
        return (;
            max_abs_relative_vol = gdf.abs_target[idx],
            name1 = titlecase(first(gdf.name1)),
            name2 = titlecase(first(gdf.name2)),
            similarity = first(gdf.similarity),
            temperature = gdf[idx, "temperature [kelvin]"]
        )
    end

    dropmissing!(df)
    transform!(df, :compound_id => ByRow(x -> "$(label_smi(x[1])) & $(label_smi(x[2]))") => :labels)

    scatter!(
        ax1, df[!, "similarity"],  df[!, "max_abs_relative_vol"];
        color = (MISTStyle.UM_COLORS.blue, 0.5), marker=:circle,
        strokewidth=0.25pt,
        strokecolor=:black,
        markersize=4pt,
    )

    df_up = subset(df,
        :similarity => ByRow(>(sim_threshold)),
        :max_abs_relative_vol => ByRow(>(excess_threshold)),
    )

    points = Point2.(df_up.similarity, df_up.max_abs_relative_vol)
    annotation!(ax1, points; text =  df_up.labels, fontsize=5pt, shrink = (2.0, 2.0))

    df_left = subset(df,
        :similarity => ByRow(<(sim_threshold)),
        :max_abs_relative_vol => ByRow(<(excess_threshold)),
    )

    points = Point2.(df_left.similarity, df_left.max_abs_relative_vol)
    annotation!(ax1, points; text =  df_left.labels, fontsize=5pt, shrink = (2.0, 2.0))
    return f
end

function plot_soap_outliers!(f, model; xaxisvisible=true)
    ax = Axis(f[1, 1];
        limits=((0, 1), (-2.4, nothing)),
        xlabel=L"$x_1$",
        ylabel=L"$V^{E}_m \;$ (cm$^3$/mol)",
        xtickformat="{:.0%}",
        xticksvisible=xaxisvisible,
        xlabelvisible=xaxisvisible,
        xticklabelsvisible=xaxisvisible,
    )
    mixtures = [
        Dict("compounds" => ["CC1COC(=O)O1", "ClC(Cl)Cl"], "temperature" => 298.15),
        Dict("compounds" => ["NCCN", "OCCO"], "temperature" => 298.15),
        Dict("compounds" => ["CN(CCO)CCO", "O"], "temperature" => 298.15),
        Dict("compounds" => ["CN1CCN(C)C1=O", "O"], "temperature" => 298.15)
    ]
    df = Mixtures.evaluate_mixtures(model, mixtures; n=50, gradients=false)
    df.x1 = first.(df.composition)

    data = DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "soap_similarity.csv")))
    subset!(data, "temperature [kelvin]" => ByRow(==(298.15)))
    mix_elems = PolyElement[]
    for gdf in groupby(df, ["compounds"])
        compounds = first(gdf.compounds)
        df_ref = subset(data,
            :smi1 => ByRow(in(compounds)),
            :smi2 => ByRow(in(compounds)),
        )
        name1, name2 = label_smi.(compounds)
        label = "$name1 & $name2"
        h = lines!(ax, gdf.x1, gdf.molar_volume_excess; linestyle=:solid, label)
        push!(mix_elems, PolyElement(; color=h.color, label=h.label))

        if nrow(df_ref)> 2
            dropmissing!(df_ref, "excess molar volume [centimeter ** 3 / mole]")
            scatter!(ax, df_ref[!, "x1"], df_ref[!, "excess molar volume [centimeter ** 3 / mole]"];
                label,
                color=h.color,
            )
        end
    end

    data_elems = [
        LineElement(; color=:black, label="MIST"),
        MarkerElement(; color=:black, label="MIST", marker=:x),
    ]

    Legend(f[1, 1],
        mix_elems,
        MISTStyle.label.(mix_elems);
        nbanks=2,
        valign=:bottom,
        halign=:right,
        titlesize=5pt,
    )
    return f, ax
end

function plot_ionic_conductivity!(f)

    cb = Colorbar(f[1, 1];
        label = L"Temperature (K)$$",
        colormap = MISTStyle.CONTINUOUS_COLORS,
        colorrange = (240, 340),
        flipaxis = false,
        vertical=false,
        tellheight=true,
        tellwidth=true,
    )

    df = DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "ionic_conductivity_curves.csv")))
    df = df[1:2:end, :]
    salts = unique(df.salt_name)
    line_styles = [:solid, :dot]
    # Sort temperatures and build a continuous, dark‐cropped colormap
    temps = sort(unique(df.temperature))

    ax = Axis(f[2, 1];
            xlabel = L"x_{Li}",
            ylabel = L"$\sigma$ (mS/cm)",
            limits= ((0, 0.20), (0, 0.35)),
            yticksvisible=true,
            yticklabelsvisible=true,
    )
    x_min = 0.160369437447523
    x_max = 0.20
    y_min = 0.0
    y_max = 0.35
    poly!(ax, Point2f[(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)];
        color = MISTStyle.UM_COLORS.maize,
        alpha=0.2,
        strokewidth = 0
    )
    # Plot each temperature curve in both panels
    for (idx, salt) in enumerate(salts), T in temps
        mask = (df.salt_name .== salt) .& (df.temperature .== T)
        sub = df[mask, :]
        label = (salt == "LiPF6") ? L"LiPF$_6$" : salt
        lines!(
            ax,
            sub.composition,
            sub.predictions;
            linestyle = line_styles[idx],
            color     = sub.temperature,
            label,
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
    poly!(
        Point2f[(x_min, y_min), (x_min, y_max), (data_T_max, y_max), (data_T_max, y_min)];
        color = MISTStyle.UM_COLORS.maize,
        alpha = 0.2,
        strokewidth = 0
    )

    return f
end

plot_thermal_alpha(model; kwargs...) = plot_thermal_alpha!(Figure(), model; kwargs...)
function plot_thermal_alpha!(f, model; xaxisvisible=true)
    mixtures = [
        ("CC#N", "C(CO)O"),     # Acetonitrile and 1,2-ethanediol
        ("CC#N", "CC(CO)O"),    # ACN and 1,2-propanediol
        ("CC#N", "C(CO)CO"),    # ACN and 1,3-propanediol
    ]
    mixtures = Dict[]
    for other in ["C(CO)O", "CC(CO)O", "C(CO)CO"]
        push!(mixtures, Dict("compounds" => ["CC#N", other], "temperature" => 298.15))
    end
    df = Mixtures.evaluate_mixtures(model, mixtures; n=32, gradients=true)
    df.x1 .= first.(df.composition)
    @. df.alpha_excess = df.molar_volume_excess_dT / df.molar_volume

    # Parse reference data
    ref = read(joinpath(DATA_DIR, "mixtures", "alpha_data.jsonc"), String)
    ref = replace(ref, r"//.*" => "")
    ref = JSON.parse(ref)
    ref = Dict(x["mixture"] => x for x in ref)

    # scale = 1e3
    scale = 1
    ax = Axis(f[1, 1];
        limits=((0,1), (-2.25e-4, 0.5e-4)),
        xlabel=L"x_1",
        xtickformat="{:.0%}",
        ylabel=L"$\alpha^{\mathrm{E}}$ (K$^{-1}$)",
        ytickformat=MISTStyle.sci_notation(),
        yticks=[0, -0.5, -1.0, -1.5] .* 1e-4,
        xminorticks=IntervalsBetween(5),
        xminorticksvisible=xaxisvisible,
        xticksvisible=xaxisvisible,
        xlabelvisible=xaxisvisible,
        xticklabelsvisible=xaxisvisible,
    )
    rel_scale = 3.3e-4
    ax_left = Axis(f[1, 1];
        limits=lift(x -> (x[1], rel_scale .* x[2]), ax.limits),
        yaxisposition=:right,
        # ylabel=L"Est. $\alpha^{\mathrm{E}}$ (K$^{-1}$)",
        ytickformat=ax.ytickformat,
        yticks=@lift($(ax.yticks) .* rel_scale),
    )
    hideydecorations!(ax_left; label=false, ticklabels=false, ticks=false)
    hidexdecorations!(ax_left)
    linkxaxes!(ax, ax_left)

    elems = PolyElement[]
    for mixture in ["ACN+PEG", "ACN+1,2-ED", "ACN+1,3-PD"]
        row = ref[mixture]
        h = scatter!(ax, row["x1"], row["alpha"]; label=replace(mixture, "+" => " & "))
        push!(elems, PolyElement(; color=h.color, label=h.label))
    end

    foreach(groupby(df, [:compounds, :temperature])) do gdf
        sort!(gdf, :x1)
        lines!(ax_left, gdf.x1, gdf.alpha_excess)
    end

    Legend(f[1, 1], elems, MISTStyle.label.(elems);
        orientation=:horizontal,
        halign=:right,
        valign=:bottom,
    )
    return f, ax
end

function label_functional_groups(smi::String)
    pyfg = @pyconst pyimport("electrolyte_fm.interpretibility.functional_groups")
    groups = pyconvert(Vector{String}, pyfg.identify_functional_groups(smi))
    length(groups) == 0 ? ["Other"] : groups
end

function plot_mixture_coverage!(f, df)
    df = select(df,
        :smi1 => ByRow(label_functional_groups) => :fg1,
        :smi2 => ByRow(label_functional_groups) => :fg2,
    )
    fg = unique(Iterators.flatten(df.fg1))
    union!(fg, Iterators.flatten(df.fg2))
    pairs = Matrix{Int}(undef, length(fg), length(fg))
    fg_count = Vector{Int}(undef, length(fg))
    for I in CartesianIndices(pairs)
        fga  = fg[I[1]]
        fgb  = fg[I[2]]
        pairs[I] = count(zip(df.fg1, df.fg2)) do (fg1, fg2)
            return (fga in fg1 && fgb in fg2) || (fga in fg2 && fgb in fg1)
        end
        if I[1] == I[2]
            fg_count[I[1]] = count(zip(df.fg1, df.fg2)) do (fg1, fg2)
                return fga in fg1 || fga in fg2
            end
        end
    end
    sdx = sortperm(fg_count; rev=true)
    pairs = pairs[sdx, sdx]
    fg = fg[sdx]
    fg_count = fg_count[sdx]

    f = GridLayout(f)
    axb = Axis(f[1, 1];
        limits = ((0, nothing), nothing),
        xticks = WilkinsonTicks(3),
        yticks = (1:length(fg), titlecase.(fg)),
        # xlabel = "Examples",
    )

    ax = Axis(f[1, 2];
        yticksvisible=false,
        yticklabelsvisible=false,
        xticks = (1:length(fg), fg),
        xticklabelrotation=pi/2,
        xticksvisible=false,
        xticklabelsvisible=false,
    )
    h = heatmap!(ax, pairs;
        colormap=Reverse(:oslo10),
        colorrange=(1, 3000),
        colorscale=log10,
        highclip=:black,
        lowclip=:white,
    )
    cb = Colorbar(f[0, :], h;
        label = "Examples",
        vertical=false,
        tellheight=true,
        tellwidth=false,
    )
    barplot!(axb, fg_count;
        direction = :x,
        strokewidth=0.25pt,
        strokecolor=:black,
        color=fg_count,
        colorscale=cb.scale,
        colormap=cb.colormap,
        highclip=cb.highclip,
        lowclip=cb.lowclip,
    )

    linkyaxes!(ax, axb)
    colgap!(f, 1, 2pt)

    return f
end

function mixture_panel(model::Py, model_strict::Py, df_excess::DataFrame)
    f = Figure(; size=(183mm, 70mm), figure_padding=(1, 1, 1, 1))

    df = DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "excess_v5.csv")))
    plot_mixture_coverage!(f[1, 1], df)
    plot_ionic_conductivity!(f[2, 1])

    gl = GridLayout(f[1:2, 2])
    _, ax_soap = plot_soap_outliers!(gl[1, 1], model; xaxisvisible=false)
    _, ax_expr = plot_experimental_data!(gl[2, 1], model; xaxisvisible=false)
    _, ax_alpha = plot_thermal_alpha!(gl[3, 1], model; xaxisvisible=true)
    linkxaxes!(ax_soap, ax_expr, ax_alpha)


    plot_soap!(f[1, 3])
    plot_excess_skewness!(f[2, 3], df_excess)

    # sublabel!(f[1, 1, TopLeft()], "c"; left=15pt)
    # sublabel!(f[1, 2, TopLeft()], "d"; left=15pt)
    # sublabel!(f[1, 3, TopLeft()], "e"; left=5pt)
    # sublabel!(f[2, 1, TopLeft()], "f"; left=15pt)
    # sublabel!(f[2, 2, TopLeft()], "g"; left=15pt)
    # sublabel!(f[2, 3, TopLeft()], "h"; left=20pt)

    colgap!(f.layout, 6pt)
    resize_to_layout!(f)

    return f
end


function mixture_panel(model_id::String, model_strict_id::String)

    # Full Binary mixture dataset
    excess_dataset = joinpath(DATA_DIR, "mixtures", "excess_v5.csv")

    # Model Trained on Random Split
    model = Mixtures.load_excess_model(joinpath(DATA_DIR, "models", model_id)).to("mps")
    df_excess = Mixtures.evaluate_binary_csv(model, excess_dataset)

    # Model Trained on Compound Split
    model_strict = Mixtures.load_excess_model(joinpath(DATA_DIR, "models", model_strict_id)).to("mps")

    return mixture_panel(model, model_strict, df_excess)
end
