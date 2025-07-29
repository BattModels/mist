using PythonCall
using DataFrames
using Makie
using CSV: CSV
using MISTStyle
using Format: format
using JSON
using StatsBase: mean, cor, corspearman, weights, tiedrank, countmap
using JSON: JSON
using Interpolations

using Mixtures: Mixtures, clean_target_name, titlecase

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

function plot_experimental_data!(f, model; xaxisvisible=true)
    ax = Axis(f[1, 1];
        limits=((0, 1), (nothing, 0.13)),
        xlabel=L"x_1",
        ylabel=L"Relative $\rho^E$",
        xtickformat="{:.0%}",
        ytickformat="{:.0%}",
        xticksvisible=xaxisvisible,
        xlabelvisible=xaxisvisible,
        xticklabelsvisible=xaxisvisible,
        tellwidth=true
    )

	mixtures = [
		Dict("compounds" => ["CC#N", "CC(=O)OCCOC(=O)C"], "temperature" => 299.25),
		Dict("compounds" => ["CC#N", "CCC1COC(=O)O1"], "temperature" => 299.25),
	]
	dfs = Mixtures.evaluate_mixtures(model, mixtures; gradients = false)
	dfs.x1 = first.(dfs.composition)
	for gdf in groupby(dfs, ["compounds"])
		lines!(ax, gdf.x1, gdf.density_excess ./ gdf.density; linestyle = :solid)
	end

	exp_dfs = [
		DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "experimental_ACN_EGDA.csv"))) => "ACN & EGDA",
		DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "experimental_ACN_BC.csv"))) => "ACN & BC",
	]

	elems = PolyElement[]
	for (df, label) in exp_dfs
		h = scatter!(ax, df.x1, df[!, "Excess Density [g/cm3]"] ./ df[!, "Density (g/cm^3)"]; label)
		push!(elems, PolyElement(; color = h.color, label = h.label))
	end
	


	Legend(f[1, 1], elems, MISTStyle.label.(elems);
		halign = :left,
		valign = :top,
		orientation = :horizontal,
	)

	return f, ax
end

plot_excess_skewness(df) = plot_excess_skewness!(Figure(), df)
function plot_excess_skewness!(f, df)
	skew = Mixtures.excess_skew(df)
	ax = Axis(f[1, 1];
		xlabel = "Reference Asymmetry",
		ylabel = "Predicted Asymmetry",
		xtickformat = "{:.0%}",
		ytickformat = "{:.0%}",
		limits = ((0, 0.5), (0, 0.5)),
		xticks = WilkinsonTicks(5),
		xminorticks = IntervalsBetween(5),
		xminorticksvisible = true,
		yticks = WilkinsonTicks(5),
		yminorticks = IntervalsBetween(5),
		yminorticksvisible = true,
		xscale = sqrt,
		yscale = sqrt,
	)

	cb = Colorbar(f[1, 2];
		label = "Max Abs. Rel. Excess",
		colormap = Reverse(MISTStyle.CONTINUOUS_COLORS),
		colorrange = (0, 0.1),
		tickformat = "{:.0%}",
		tellheight = true,
		minorticksvisible = true,
		minorticks = IntervalsBetween(2),
		flip_vertical_label = true,
		vertical = true,
	)
	targets = [
		("Molar Volume", "molar_volume", :circle),
		("Density", "density", :rect),
		("Molar Enthalpy", "molar_enthalpy", :utriangle),
	]
	for (label, target, marker) in targets
		df_target = select(skew, "compound_id", target, "$(target)_ref", "$(target)_rel_ref")
		dropmissing!(df_target)
		transform!(df_target, "$(target)_rel_ref" => ByRow(abs) => "$(target)_rel_ref")
		sort!(df_target, "$(target)_rel_ref"; rev = false)
		x = df_target[!, "$(target)_ref"]
		y = df_target[!, target]
		color = df_target[!, "$(target)_rel_ref"]
		nrow(df_target) == 0 && continue
		scatter!(ax, x, y;
			label,
			marker,
			color,
			alpha=0.6,
			MISTStyle.cb_attrs(cb, Scatter)...,
		)

		# Compute and report correlations
		weight = weights(color)
		pearson = cor(x, y)
		pearson_weighted = cor(hcat(x, y), weight)[1, end]
		spearman = corspearman(x, y)
		spearman_weighted = cor(hcat(tiedrank(x), tiedrank(y)), weight)[1, end]
		@info "Excess Skew Correlations: $target" pearson spearman pearson_weighted spearman_weighted
	end

	margin = something(theme(:Legend), (; margin = (1, 1, 1, 1))).margin
	axislegend(ax; position = :lt, orientation = :horizontal, margin)
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
		limits = ((x_min, 1), (0, y_max)),
		xlabel = "ReMATCH Similarity",
		ylabel = L"Max $\left| V^{E}_m \right|$",
		xtickformat = "{:.0%}",
		yticks = WilkinsonTicks(3; k_min = 3, k_max = 5),
	)
	df = DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "soap_similarity.csv")))

	df[!, :abs_target] = abs.(
		df[!, :"excess molar volume [centimeter ** 3 / mole]"] ./
		df[!, :"molar volume [centimeter ** 3 / mole]"],
	)
	transform!(df, ["smi1", "smi2"] => ByRow((x, y) -> sort([x, y])) => :compound_id)
	df = combine(groupby(df, [:compound_id])) do gdf
		idx = argmax(gdf.abs_target)
		return (;
			max_abs_relative_vol = gdf.abs_target[idx],
			name1 = titlecase(first(gdf.name1)),
			name2 = titlecase(first(gdf.name2)),
			similarity = first(gdf.similarity),
			temperature = gdf[idx, "temperature [kelvin]"],
		)
	end

	dropmissing!(df)
	transform!(df, :compound_id => ByRow(x -> "$(label_smi(x[1])) & $(label_smi(x[2]))") => :labels)

	scatter!(
		ax1, df[!, "similarity"], df[!, "max_abs_relative_vol"];
		color = (MISTStyle.UM_COLORS.blue, 0.5), marker = :circle,
		strokewidth = 0.25pt,
		strokecolor = :black,
		markersize = 4pt,
	)

	df_up = subset(df,
		:similarity => ByRow(>(sim_threshold)),
		:max_abs_relative_vol => ByRow(>(excess_threshold)),
	)

	points = Point2.(df_up.similarity, df_up.max_abs_relative_vol)
	annotation!(ax1, points; text = df_up.labels, fontsize = 5pt, shrink = (2.0, 2.0))

	df_left = subset(df,
		:similarity => ByRow(<(sim_threshold)),
		:max_abs_relative_vol => ByRow(<(excess_threshold)),
	)

	points = Point2.(df_left.similarity, df_left.max_abs_relative_vol)
	annotation!(ax1, points; text = df_left.labels, fontsize = 5pt, shrink = (2.0, 2.0))
	return f
end

function plot_soap_outliers!(f, model; xaxisvisible = true)
	ax = Axis(f[1, 1];
		limits = ((0, 1), (-2.5, nothing)),
		xlabel = L"$x_1$",
		ylabel = L"$V^{E}_m \;$ (cm$^3$/mol)",
		xtickformat = "{:.0%}",
		xticksvisible = xaxisvisible,
		xlabelvisible = xaxisvisible,
		xticklabelsvisible = xaxisvisible,
	)
	mixtures = [
		Dict("compounds" => ["CC1COC(=O)O1", "ClC(Cl)Cl"], "temperature" => 298.15),
		Dict("compounds" => ["NCCN", "OCCO"], "temperature" => 298.15),
		Dict("compounds" => ["CN(CCO)CCO", "O"], "temperature" => 298.15),
		Dict("compounds" => ["CN1CCN(C)C1=O", "O"], "temperature" => 298.15),
	]
	df = Mixtures.evaluate_mixtures(model, mixtures; n = 50, gradients = false)
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
		h = lines!(ax, gdf.x1, gdf.molar_volume_excess; linestyle = :solid, label)
		push!(mix_elems, PolyElement(; color = h.color, label = h.label))

		if nrow(df_ref) > 2
			dropmissing!(df_ref, "excess molar volume [centimeter ** 3 / mole]")
			scatter!(ax, df_ref[!, "x1"], df_ref[!, "excess molar volume [centimeter ** 3 / mole]"];
				label,
				color = h.color,
			)
		end
	end

	data_elems = [
		LineElement(; color = :black, label = "MIST"),
		MarkerElement(; color = :black, label = "MIST", marker = :x),
	]

	Legend(f[1, 1],
		mix_elems,
		MISTStyle.label.(mix_elems);
		nbanks = 2,
		valign = :bottom,
		halign = :right,
		titlesize = 5pt,
	)
	return f, ax
end

function plot_ionic_conductivity!(f)

	cb = Colorbar(f[1, 2];
		label = L"Temperature (K)$$",
		colormap = MISTStyle.CONTINUOUS_COLORS,
		colorrange = (240, 340),
		vertical = true,
		tellheight = true,
		tellwidth = true,
		flipaxis = true,
		flip_vertical_label = true,
	)

	df = DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "ionic_conductivity_curves.csv")))
	df = df[1:2:end, :]
	salts = unique(df.salt_name)
	line_styles = [:solid, :dot]
	# Sort temperatures and build a continuous, dark‐cropped colormap
	temps = sort(unique(df.temperature))

	ax = Axis(f[1, 1];
		xlabel = L"x_{Li}",
		ylabel = L"$\sigma$ (mS/cm)",
		limits = ((0, 0.20), (0, 0.35)),
		yticksvisible = true,
		yticklabelsvisible = true,
	)
	x_min = 0.160369437447523
	x_max = 0.20
	y_min = 0.0
	y_max = 0.35
	poly!(ax, Point2f[(x_min, y_min), (x_min, y_max), (x_max, y_max), (x_max, y_min)];
		color = MISTStyle.UM_COLORS.maize,
		alpha = 0.2,
		strokewidth = 0,
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
			MISTStyle.cb_attrs(cb, Lines)...,
		)
	end
	axislegend(ax, position = :rt, padding = (1, 1, 1, 1), margin = (1, 1, 1, 1), unique = true)
	return f
end

plot_thermal_alpha(model; kwargs...) = plot_thermal_alpha!(Figure(), model; kwargs...)
function plot_thermal_alpha!(f, model; xaxisvisible = true)
	mixture_map = Dict(
		["CC#N", "C(CO)O"] => "ACN+1,2-ED",     # Acetonitrile and 1,2-ethanediol
		["CC#N", "CC(CO)O"] => "ACN+PEG",    # ACN and 1,2-propanediol
		["CC#N", "C(CO)CO"] => "ACN+1,3-PD",    # ACN and 1,3-propanediol
	)
	mixtures = Dict[]
	for other in ["C(CO)O", "CC(CO)O", "C(CO)CO"]
		push!(mixtures, Dict("compounds" => ["CC#N", other], "temperature" => 298.15))
	end
	df = Mixtures.evaluate_mixtures(model, mixtures; n = 50, gradients = true)
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
		limits = ((0, 1), (-2.25e-4, 0.5e-4)),
		xlabel = L"x_1",
		xtickformat = "{:.0%}",
		ylabel = L"$\alpha^{\mathrm{E}}$ (K$^{-1}$)",
		ytickformat = MISTStyle.sci_notation(),
		yticks = [0, -0.5, -1.0, -1.5] .* 1e-4,
		xminorticks = IntervalsBetween(5),
		xminorticksvisible = xaxisvisible,
		xticksvisible = xaxisvisible,
		xlabelvisible = xaxisvisible,
		xticklabelsvisible = xaxisvisible,
	)

	result_scale = 0
	foreach(groupby(df, [:compounds, :temperature])) do gdf
		sort!(gdf, :x1)
		row = ref[mixture_map[first(gdf.compounds)]]
		itp = LinearInterpolation(gdf.x1, gdf.alpha_excess )
		result_scale += mean(itp(row["x1"])./row["alpha"])
	end
	rel_scale = result_scale/3

	ax_left = Axis(f[1, 1];
		limits = lift(x -> (x[1], rel_scale .* x[2]), ax.limits),
		yaxisposition = :right,
		ylabel = L"Est. $\alpha^{\mathrm{E}}$ (K$^{-1}$)",
		ytickformat = ax.ytickformat,
		flip_ylabel = true,
		yticks = @lift($(ax.yticks) .* rel_scale),
	)
	hideydecorations!(ax_left; label = false, ticklabels = false, ticks = false)
	hidexdecorations!(ax_left)
	linkxaxes!(ax, ax_left)

	elems = PolyElement[]
	for mixture in ["ACN+PEG", "ACN+1,2-ED", "ACN+1,3-PD"]
		row = ref[mixture]
		h = scatter!(ax, row["x1"], row["alpha"]; label = replace(mixture, "+" => " & "))
		push!(elems, PolyElement(; color = h.color, label = h.label))
	end

	foreach(groupby(df, [:compounds, :temperature])) do gdf
		sort!(gdf, :x1)
		lines!(ax_left, gdf.x1, gdf.alpha_excess)
	end

	Legend(f[1, 1], elems, MISTStyle.label.(elems);
		orientation = :horizontal,
		halign = :right,
		valign = :bottom,
	)
	return f, ax
end

function plot_diffmix_enthalpy!(f, model)
	col = "molar_enthalpy_excess"
	smiles_to_name = Dict(
		"CC1COC(=O)O1" => "PC", 
		"CCCCC(=O)OCC" => "EP", # Ethyl pentanoate
		"CCCOC(=O)CCC" => "PP", # Propyl Propanoate
		"CC(=O)OCC(C)C" => "2-MA", # 2-methylpropyl acetate 
		"CS(C)=O" => "DMS", # Dimethyl Sulfoxide
		"CCOc1ccccc1" => "Phenetole",
		"CCOc1ccccc1" => "Anisole",
		"CCCCCCC(=O)OCC" => "EH", # Ethyl Heptanoate
		"CCCCC(=O)OC" => "MP", # Methy Petanoate
		"CCCCOC(=O)CCC" => "Butanoic", # Butonic acid
		"CC(=O)OC(C)(C)C" => "Acetic",
		"CCCCCCCC(=O)OCC" => "Octanoic",
		"CCOC(=O)C(C)C" => "Propanoic",
		"CCCCCC(=O)OCC" => "Hexanoic",
		"CCOC(=O)OCC" => "EC",
		"CCCOC(=O)CC" => "PE", # Propyl Ester
		"COC(=O)OC" => "DMC"
	)
	
	excess_dataset = joinpath(DATA_DIR, "mixtures", "diffmix_enthalpy.csv")
	df_excess = Mixtures.evaluate_binary_csv(model, excess_dataset)
	df_excess = subset(df_excess,
		:temperature => ByRow(<(300.)),
	)
	@info unique(first.(df_excess.temperature))
	@info unique(last.(df_excess.compounds))
	ax2 = Axis(f[1, :];
		limits = ((0, 1), (0, nothing)),
		xlabel = L"x_1",
		ylabel =    L"$H_m$",
		tellwidth = true,
	)

    df_excess.x1 = first.(df_excess.composition)
	df_excess.smi2 = last.(df_excess.compounds)
	sort!(df_excess, :smi2)

	counter = 0
	elems = PolyElement[]
    for gdf in groupby(df_excess, [ "compounds"])
		if counter > 3
			break
		end
		name1 = smiles_to_name[first(gdf.compounds)[1]]
		name2 = smiles_to_name[first(gdf.compounds)[2]]
		label = "$name2"
		h = lines!(ax2, gdf.x1, gdf[!, "$col"]; 
            linestyle = :solid,
			label,
        )
        
        scatter!(ax2, gdf.x1, gdf[!, "$(col)_ref"];
        )
		push!(elems, PolyElement(; color = h.color, label = h.label))
		counter += 1
	end
	Legend(f[1, 1], elems, MISTStyle.label.(elems);
		orientation = :horizontal,
		halign = :center,
		valign = :bottom,
		nbanks=2
	)
    return f, ax2
end

function mixture_panel(model::Py, model_strict::Py, df_excess::DataFrame)

	f = Figure(; size = (183mm, 72mm), figure_padding = (2, 2, 2, 2))

    df = DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "excess_v6.csv")))
    df = Mixtures.normalize_dataset(df)

    gl = GridLayout(f[1:3, 1:2])
    Mixtures.plot_mixture_coverage!(gl[1, 1], df)
    plot_excess_skewness!(gl[2, 1], df_excess)
    plot_ionic_conductivity!(gl[2, 2])
    plot_soap!(gl[1, 2])
    sublabel!(gl[1, 1, TopLeft()], "c"; left=15pt)
    sublabel!(gl[2, 1, TopLeft()], "d"; left=15pt)
    sublabel!(gl[1, 2, TopLeft()], "e"; left=5pt)
    sublabel!(gl[2, 2, TopLeft()], "f"; left=5pt)

    gl = GridLayout(f[:, 3])
	_, ax_diffmix = plot_diffmix_enthalpy!(gl[3, :], model)
    _, ax_soap = plot_soap_outliers!(gl[1,  :], model; xaxisvisible=false)
    _, ax_expr = plot_experimental_data!(gl[2,  :], model_strict; xaxisvisible=false)
	linkxaxes!(ax_diffmix, ax_soap, ax_expr)

    sublabel!(gl[1, 1, TopLeft()], "g"; left=15pt)
    sublabel!(gl[2, 1, TopLeft()], "h"; left=15pt)
    sublabel!(gl[3, 1, TopLeft()], "i"; left=20pt)

    colgap!(f.layout, 6pt)
    resize_to_layout!(f)

	return f
end


function mixture_panel(model_id::String, model_strict_id::String)

    # Full Binary mixture dataset
    excess_dataset = joinpath(DATA_DIR, "mixtures", "excess_dataset_v6", "random")
	# Model Trained on Random Split
	model = Mixtures.load_excess_model(joinpath(DATA_DIR, "models", model_id)).to("mps")
	df_excess = Mixtures.evaluate_dataset(model, excess_dataset)

	# Model Trained on Compound Split
	model_strict = Mixtures.load_excess_model(joinpath(DATA_DIR, "models", model_strict_id)).to("mps")

	return mixture_panel(model, model_strict, df_excess)
end
