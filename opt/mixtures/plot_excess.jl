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

const DATA_DIR = realpath(joinpath(pkgdir(Mixtures), "..", "..", "data"))

markers_labels = [
    (:circle, ":circle"),
    (:rect, ":rect"),
    (:diamond, ":diamond"),
    (:hexagon, ":hexagon"),
    (:cross, ":cross"),
    (:xcross, ":xcross"),
    (:utriangle, ":utriangle"),
    (:dtriangle, ":dtriangle"),
    (:ltriangle, ":ltriangle"),
    (:rtriangle, ":rtriangle"),
    (:pentagon, ":pentagon"),
    (:star4, ":star4"),
    (:star5, ":star5"),
    (:star6, ":star6"),
    (:star8, ":star8"),
    (:vline, ":vline"),
    (:hline, ":hline"),
]


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
	cb = Colorbar(gl_parity[1, 2]; colormap = :managua, colorrange = extrema(df.temperature), flip_vertical_label = true)
	temperature = df[!, "temperature"]
	inspector_columns = ["temperature", "compounds", "composition"]
	for (tdx, target) in enumerate(targets)
		ax_y = Axis(gl[1, tdx])
		ablines!(ax_y, 0, 1; color = :black, linestyle = :dash)
		scatter!(ax_y, df[!, "$(target)_ref"], df[!, target];
			color = temperature,
			inspectable = true,
			inspector_label = di(df, mixture_selection, [target, "$(target)_ref", inspector_columns...]),
			MISTStyle.cb_attrs(cb, Scatter)...,
		)

		ax_e = Axis(gl[2, tdx]; xlabel = target)
		ablines!(ax_e, 0, 1; color = :black, linestyle = :dash)
		scatter!(ax_e, df[!, "$(target)_excess_ref"], df[!, "$(target)_excess"];
			color = temperature,
			inspectable = true,
			inspector_label = di(df, mixture_selection, [target, "$(target)_ref", inspector_columns...]),
			MISTStyle.cb_attrs(cb, Scatter)...,
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
		ax = Axis(gl_predict[1, tdx]; limits = ((0, 1), extrema(df[!, target])))
		h = scatter!(ax, lift_points(dfs, "x1", target); color = :red)
		h = scatter!(ax, lift_points(dfs, "x1", "$(target)_ref"); color = :blue)
		on(_ -> autolimits!(ax), mixture_selection)

		ax = Axis(gl_predict[2, tdx]; limits = ((0, 1), nothing))
		scatter!(ax, lift_points(dfs, "x1", "$(target)_excess"); color = :red)
		scatter!(ax, lift_points(dfs, "x1", "$(target)_excess_ref"); color = :blue)
		on(_ -> autolimits!(ax), mixture_selection)
	end
	predict_label = lift(mixture_selection) do (smi1, smi2, temp)
		return "$smi1 vs. $smi2 at $(format("{:.2f}", temp)) K"
	end
	Label(gl_predict[0, :], predict_label)

	elems = [
		LineElement(; color = :red, label = "Prediction"),
		LineElement(; color = :blue, label = "Reference"),
	]
	Legend(gl_predict[3, :], elems, MISTStyle.label.(elems);
		nbanks = 2,
		tellheight = true,
		tellwidth = false,
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
	smi1 = Menu(ui[1, 1], options = compounds)

	smi2_options = lift(smi1.selection) do smi1
		dfs = subset(df, :compounds => ByRow(x -> smi1 in x))
		unique(Iterators.flatten(dfs.compounds))
	end
	smi2 = Menu(ui[1, 2], options = smi2_options)

	df = transform(df, :composition => ByRow(first) => :x1)

	dfs = lift(smi1.selection, smi2.selection) do smi1, smi2
		subset(df, :compounds => ByRow(x -> smi1 in x && smi2 in x))
	end

	for (tdx, target) in enumerate(targets)
		ax = Axis(gl[1, tdx]; limits = ((0, 1), nothing))
		scatter!(ax, lift_points(dfs, "x1", target); label = "Prediction")
		scatter!(ax, lift_points(dfs, "x1", "$(target)_ref"); label = "Reference")
		on(_ -> autolimits!(ax), smi2.selection)

		ax = Axis(gl[2, tdx]; limits = ((0, 1), nothing))
		scatter!(ax, lift_points(dfs, "x1", "$(target)_excess"))
		scatter!(ax, lift_points(dfs, "x1", "$(target)_excess_ref"))
		on(_ -> autolimits!(ax), smi2.selection)
	end

	notify(smi1.selection)
	notify(smi2.selection)


	return f
end

function plot_ionic_temperature!(f)
	ax = Axis(f[1, 1];
		xlabel = "Temperature (K)",
		ylabel = L"$\sigma$ (mS/cm)",
	)
	df = DataFrame(CSV.File("panel_data/temperature_w8phe03q.csv"))
	errorbars!(ax, df[!, "temperature"], df[!, "predicted"], df[!, "st_dev"], color = :black)
	scatter!(ax, df[!, "temperature"], df[!, "predicted"], markersize = 3)
	data_T_max = 293.150000
	x_min, x_max = extrema(df[!, "temperature"])
	y_min, y_max = extrema(df[!, "predicted"])
	poly!(
		Point2f[(x_min, y_min), (x_min, y_max), (data_T_max, y_max), (data_T_max, y_min)];
		color = MISTStyle.UM_COLORS.maize,
		alpha = 0.2,
		strokewidth = 0,
	)

	return f
end

metadata = Dict(
    "diffmix/excess_molar_volume.csv" => [L"$V_m^E$ (cm$^3$/mol)", "molar_volume_excess", false],
    "diffmix/excess_molar_enthalpy.csv" => [L"$H_m^E$ (cm$^3$/mol)", "molar_enthalpy_excess", true],
)

function diffmix_validation(model_id)
    f = Figure(; figure_padding=(2,2,2,2))
    cb = Colorbar(f[3, 1:2];
		label = "Temperature (K)",
        colormap =  :managua,
        colorrange = (298, 309),
		tellheight = true,
        tellwidth = true,
		vertical = false,
        flip_vertical_label=true
	)

    for (idx, excess_dataset) in enumerate([ "diffmix/excess_molar_volume.csv", "diffmix/excess_molar_enthalpy.csv"])
        diffmix_validation!(model_id, f[idx, :], excess_dataset, cb)
    end
    return f
end

function diffmix_validation!(model_id, f, excess_dataset, cb)
    quantity = metadata[excess_dataset][1]
    col = metadata[excess_dataset][2]
    model = Mixtures.load_excess_model(joinpath(DATA_DIR, "models", model_id)).to("mps")
	df_excess = Mixtures.evaluate_binary_csv(model, excess_dataset)
    df_ref = load_reference(excess_dataset)
    ax1 = Axis(f[1, 1];
		xlabel = "Exp",
		ylabel =  "MIST",
	)

    scatter!(ax1, df_excess[!, "$(col)_ref"], df_excess[!, "$col"];
        color = df_excess[!, "temperature"],
        MISTStyle.cb_attrs(cb, Scatter)...,
	)
    lines!(ax1, df_excess[!, "$(col)_ref"], df_excess[!, "$(col)_ref"];
        color=:black
    )
    ax2 = Axis(f[1, 2];
		limits = ((0, 1), nothing),
		xlabel = L"x_1",
		ylabel =    quantity,
		tellwidth = true,
	)
    if length(df_excess.composition) > 500
        df_excess = df_excess[1: 1: 500, :]
    end
    df_excess.x1 = first.(df_excess.composition)
    marker_lookup =  Dict(zip(unique(df_excess.compounds), first.(markers_labels)))
    for gdf in groupby(df_excess, ["compounds", "temperature"])
		lines!(ax2, gdf.x1, gdf[!, "$col"];
            linestyle = :solid,
            color     = gdf.temperature,
            MISTStyle.cb_attrs(cb, Lines)...,
        )

        scatter!(ax2, gdf.x1, gdf[!, "$(col)_ref"];
            color     = gdf.temperature,
            marker = marker_lookup[first(gdf.compounds)],
            strokewidth = 0.25pt,
			strokecolor = MISTStyle.UM_COLORS.ash,
            MISTStyle.cb_attrs(cb, Scatter)...,
        )
	end
    return f
end
