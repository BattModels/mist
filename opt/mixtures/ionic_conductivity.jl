using DataFrames
using Makie
using JSON
using CSV: CSV
using MISTStyle: MISTStyle, label
using Format: format
using Statistics: mean
using Mixtures: Mixtures
using Printf
using ColorSchemes
using Combinatorics
using CategoricalArrays: categorical, levelcode

const DATA_DIR = realpath(joinpath(pkgdir(Mixtures), "..", "..", "data"))
const similarity_data = JSON.parsefile("solvent_rematch.json")
const solvents = [
    ["CC1COC(=O)O1", "O=C1OCCO1", "O=C1OCC(F)O1"],
    ["CC1COC(=O)O1", "CCOC(=O)OCC", "O=C1OCC(F)O1"],
    ["CCOC(=O)OC", "O=C1OCCO1", "O=C1OCC(F)O1"],
    ["CC1COC(=O)O1", "CCOC(=O)OC", "O=C1OCC(F)O1"],
]
const salts = ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "F[P-](F)(F)(F)(F)F"]


function label_smi(smi::AbstractString)
    known = Dict(
        "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F" => "TFSI",
        "F[P-](F)(F)(F)(F)F" => "PF6",
        "O=C1OCC(F)O1" => "FEC",
        "CC1COC(=O)O1" => "PC",
        "CCOC(=O)OC" => "EMC",
        "CCOC(=O)OCC" => "DEC",
        "O=C1OCCO1" => "EC",
        "COC(=O)OC" => "DMC",
    )
    return get(known, smi, smi)
end

function get_similarity(smiles1::AbstractString, smiles2::AbstractString)

    key1 = "$(smiles1)_$(smiles2)"
    key2 = "$(smiles2)_$(smiles1)"

    if haskey(similarity_data, key1)
        return similarity_data[key1]
    elseif haskey(similarity_data, key2)
        return similarity_data[key2]
    else
        return nothing
    end
end

function weighted_ternary_similarity(solvent_list, compositions)
    weighted_sum = 0.0

    for i in 1:length(solvent_list)
        for j in (i+1):length(solvent_list)
            sim = get_similarity(solvent_list[i], solvent_list[j])
            weighted_sum += (compositions[i] + compositions[j]) * sim
        end
    end

    return weighted_sum
end

function calculate_excess(model, mixture::Dict, salt_comp::Float64, n::Integer)

    df = Mixtures.evaluate_conductivity(model, [mixture]; n = n, fixed_salt = salt_comp)

    comp_matrix = reduce(hcat, df.composition)'
    # Update salt composition to account for discretization
    salt_comp = first(comp_matrix[:, 4])
    pure_solvent_param = Dict(
        "Ea" => zeros(Float64, 3),
        "Tg" => zeros(Float64, 3),
    )
    pure_solvent_comp = 1 - salt_comp

    for (idx, solvent) in enumerate(mixture["solvents"])
        comp = zeros(4)
        comp[idx] = pure_solvent_comp
        comp[4] = salt_comp
        preds = Mixtures.evaluate_at_composition(model, mixture, comp)
        pure_solvent_param["Ea"][idx] = preds["Ea"]
        pure_solvent_param["Tg"][idx] = preds["Tg"]
    end


    solvent_comps = comp_matrix[:, 1:3]
    solvent_fractions = solvent_comps ./ sum(solvent_comps, dims = 2)
    for parameter in ["Ea", "Tg"]
        pure_solvent_values = pure_solvent_param[parameter]
        df[!, "ideal_mixing_$(parameter)"] = solvent_fractions * pure_solvent_values
        df[!, "excess_$(parameter)"] = df[!, parameter] .- df[!, "ideal_mixing_$(parameter)"]
        df[!, "relative_excess_$(parameter)"] = abs.(df[!, "excess_$(parameter)"] ./ df[!, parameter])
        @assert sum(abs.(collect(df[1:4, "excess_$(parameter)"]))) < 1e-3 "Zero excess for expected degenerate case"
    end
    return df
end

function conductivity_composition_curve(model, mixture; n = 50)
    x1 = 1.0
    x2 = 0.0
    x3 = 0.0
    composition = range(0.02, stop = 0.20, length = n)
    conductivity = [
        Mixtures.evaluate_at_composition(model, mixture, [x1 - x, x2, x3, x])["conductivity"] for x in
                                                                                                  composition
    ]
    return composition, conductivity
end

function plot_composition_curves(model, solvent)

    temperatures = [260, 280, 300, 320, 340]
    f = Figure(; size = (61mm, 36mm), figure_padding = (2, 2, 2, 2))
    cb = Colorbar(f[1, 2];
        label = L"Temperature (K)$$",
        colormap = MISTStyle.CONTINUOUS_COLORS,
        colorrange = extrema(temperatures),
        vertical = true, tellheight = true, tellwidth = true,
        flipaxis = true, flip_vertical_label = true,
    )

    line_styles = [:solid, :dot]

    ax = Axis(f[1, 1];
        xlabel = L"x_{Li}", ylabel = L"$\sigma$ (mS/cm)",
        limits = ((0, 0.20), (0, nothing)),
    )

    for (idx, salt) in enumerate(salts), T in temperatures
        label = (label_smi(salt) == "PF6") ? L"LiPF$_6$" : L"$$LiTFSI"
        mixture = Dict(
            "solvents" => solvent,
            "temperature" => T,
            "salt" => [salt, "[Li+]"],
        )
        composition, conductivity = conductivity_composition_curve(model, mixture)
        lines!(
            ax,
            composition,
            exp.(conductivity);
            linestyle = line_styles[idx],
            color     = T,
            label,
            MISTStyle.cb_attrs(cb, Lines)...,
        )
    end

    xlims = ax.limits[][1]  # (xmin, xmax)
    ylims = (0, 10)  # (ymin, ymax)
    x_min = 0.160369437447523 # Maximum composition in training data
    poly!(ax, Point2f[(x_min, ylims[1]), (x_min, ylims[2]), (xlims[2], ylims[2]), (xlims[2], ylims[1])];
        color = MISTStyle.UM_COLORS.maize, alpha = 0.2, strokewidth = 0,
    )

    axislegend(ax, position = :rt, padding = (1, 1, 1, 1), margin = (1, 1, 1, 1), unique = true)
    return f
end

function plot_composition_curves()

    model_id = "mist-conductivity-27.0M-2mpg8dcd"
    model = Mixtures.load_conductivity_model(joinpath(DATA_DIR, "models", model_id)).to("mps")
    for solvent in solvents
        fn_name = "cond_curve_$(label_smi(solvent[1]))_$(label_smi(solvent[2]))_$(label_smi(solvent[3]))"
        with_theme(MISTStyle.theme()) do
            plot_composition_curves(model, solvent)
        end |> MISTStyle.savefig(fn_name)
    end
end

function calculate_delta_ea(model, solvent, temperatures, salts, salt_comps)
    mixtures = [
        Dict("solvents" => solvent, "temperature" => temp, "salt" => [salt, "[Li+]"])
        for temp in temperatures for salt in salts
    ]
    all_data = []

    for salt_comp in salt_comps
        df = Mixtures.evaluate_conductivity(model, mixtures; n = 25, fixed_salt = salt_comp)
        df[!, "salt_mol"] = last.(df.composition)
        df[!, "salt_anion"] = [comp[4] for comp in df.components]
        df[!, "Ea_over_T"] = df.Ea ./ df.temperature

        for temp in temperatures
            temp_df = filter(row -> row.temperature == temp, df)
            temp_gdf = groupby(temp_df, :salt_anion)
            groups = collect(temp_gdf)
            lipf6_idx = findfirst(g -> g[1, :salt_anion] == "F[P-](F)(F)(F)(F)F", groups)
            litfsi_idx = findfirst(g -> g[1, :salt_anion] != "F[P-](F)(F)(F)(F)F", groups)

            delta_Ea_over_T = groups[lipf6_idx].Ea_over_T - groups[litfsi_idx].Ea_over_T
            push!(all_data, (salt_comp = salt_comp, temperature = temp,
                delta_Ea_over_T = delta_Ea_over_T))
        end
    end
    return all_data
end

function plot_composition!(ax, all_data, comp, temperatures, dodge_width)

    n_temps = length(temperatures)
    colorscheme = ColorSchemes.colorschemes[MISTStyle.CONTINUOUS_COLORS]

    for (t_idx, temperature) in enumerate(temperatures)
        comp_temp_data = filter(d -> d.salt_comp == comp && d.temperature == temperature, all_data)
        isempty(comp_temp_data) && continue

        values = vcat([d.delta_Ea_over_T for d in comp_temp_data]...)
        position = comp + (t_idx - (n_temps+1)/2) * dodge_width
        positions = fill(position, length(values))

        temp_range = maximum(temperatures) - minimum(temperatures)
        color_val = temp_range > 0 ? (temperature - minimum(temperatures)) / temp_range : 0.5
        box_color = get(colorscheme, color_val)

        boxplot!(ax, positions, values;
            width = dodge_width * 0.7, color = (box_color, 0.9),
            strokecolor = :black, strokewidth = 0.2, show_outliers = false,
        )
    end
end

function plot_delta_Ea(model)

    temperatures = [260, 280, 300, 320, 340]
    salt_comps = range(start = 0.05, stop = 0.2, length = 5)

    fig = Figure(size = (95mm, 100mm), figure_padding = (2, 2, 2, 2))
    cb = Colorbar(fig[length(solvents)+1, 1];
        label = L"$$Temperature [K]",
        colormap = MISTStyle.CONTINUOUS_COLORS, colorrange = extrema(temperatures),
        tellheight = true, tellwidth = false, flipaxis = false, vertical = false,
    )

    for (idx, solvent) in enumerate(solvents)
        ax = Axis(
            fig[idx, 1],
            title = "$(label_smi(solvent[1])) | $(label_smi(solvent[2])) |  $(label_smi(solvent[3]))",
            xlabel = idx == length(solvents) ? L"$x_{Li^+}$" : "",
            ylabel = L"$\frac{E_{a,LiPF_6}}{T} - \frac{E_{a,LiTFSI}}{T}$",
            xticksvisible = idx==length(solvents),
            xticklabelsvisible = idx==length(solvents),
            titlesize = 6pt,
            xgridvisible = false,
            ygridvisible = false,
        )

        all_data = calculate_delta_ea(model, solvent, temperatures, salts, salt_comps)
        comp_positions = unique([d.salt_comp for d in all_data])
        dodge_width = (comp_positions[2] - comp_positions[1]) * 0.5 / length(temperatures)

        for comp in comp_positions
            plot_composition!(ax, all_data, comp, temperatures, dodge_width)
        end

        rowgap!(fig.layout, 2)
    end

    return fig
end


function plot_delta_Ea()
    model_id = "mist-conductivity-27.0M-2mpg8dcd"
    model = Mixtures.load_conductivity_model(joinpath(DATA_DIR, "models", model_id)).to("mps")
    with_theme(MISTStyle.theme()) do
        plot_delta_Ea(model)
    end |> MISTStyle.savefig("delta_Ea")
    return
end


function plot_angell_solvents()
    df = DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "ionic_conductivity_inference.csv")))
    fig = Figure(size = (53mm, 32mm), figure_padding = (2, 2, 2, 2))

    ax = Axis(fig[1, 1],
        xlabel = L"T_g/T",
        ylabel = L"ln $\sigma$",
    )

    unique_solvents = unique(df[!, "Solvent"])
    n_solvents = length(unique_solvents)
    colors = MISTStyle.CAT_COLORS[1:n_solvents]

    for (i, solvent) in enumerate(unique_solvents)
        mask = df[!, "Solvent"] .== solvent
        scatter!(ax, df[mask, "Tg/T"], df[mask, "conductivity"],
            color = colors[i],
            marker = :circle,
            markersize = 2,
            alpha = 0.6,
            label = string(solvent),
        )
    end

    Legend(fig[1, 2], ax, framevisible = true)
    return fig
end


function plot_ternary_vft_parameter!(
    ax::Axis, df::DataFrame, col_name::String,
    solvent_names::Vector{String}, global_crange::NTuple{2, Real},
)

    comp_matrix = reduce(hcat, df.composition)'
    a = comp_matrix[:, 1]
    b = comp_matrix[:, 2]
    c = comp_matrix[:, 3]

    values = df[!, col_name]
    Mixtures.ternary!(ax, a, b, c, values,
        colormap = MISTStyle.CONTINUOUS_COLORS,
        colorrange = global_crange,
        label_a = label_smi(solvent_names[1]),
        label_b = label_smi(solvent_names[2]),
        label_c = label_smi(solvent_names[3]),
        show_ticks = true)

end

function plot_ternary_vft_parameter(
    model, mixture::Dict; parameter::String = "Ea", excess::Bool = false, n::Integer = 64,
)

    fig = Figure(; size = (183mm, 55mm), figure_padding = (1, 1, 1, 1))

    solvent_names = mixture["solvents"]
    mixtures = [mixture]

    all_dfs = []
    for salt_comp in [0.05, 0.1, 0.15]
        if excess
            df = calculate_excess(model, mixture, salt_comp, n)
        else
            df = Mixtures.evaluate_conductivity(model, mixtures; n = n, fixed_salt = salt_comp)
        end
        push!(all_dfs, df)
    end
    col_name = excess ? "excess_$(parameter)" : parameter

    all_ea = vcat([df[!, col_name] for df in all_dfs]...)

    global_crange = extrema(all_ea)

    for (idx, df) in enumerate(all_dfs)
        ax = Axis(
            fig[1, idx];
            aspect = DataAspect(),
            tellwidth = true,
            tellheight = true,
            limits = ((-0.1, 1.1), (-0.2, 1.0)),
        )
        plot_ternary_vft_parameter!(ax, df, col_name, solvent_names, global_crange)
        salt_comp = first(last.(df.composition))
        hidedecorations!(ax)
        hidespines!(ax)

        val = @sprintf("%.2f", salt_comp)
        label = L"$x_{\mathrm{Li}^{+}}$ = %$val"
        Label(
            fig[0, idx], label;
            tellheight = true, fontsize = 8pt, lineheight = 0.1, padding = (0, 0, 0, 0),
        )
    end
    # Note: parameter T_0 is mislabelled Tg in model code
    label = parameter == "Tg" ? L"T_0" : L"$E_a$"
    label = excess ? L"excess %$(label) [K]" : L"%$(label) [K]"

    Colorbar(fig[1, 4],
        colormap = MISTStyle.CONTINUOUS_COLORS,
        limits = global_crange,
        label = label,
        height = Relative(0.8),
        tellheight = true,
        flip_vertical_label = true,
        vertical = true,
    )

    colsize!(fig.layout, 1, Aspect(1, 1.0))
    colsize!(fig.layout, 2, Aspect(1, 1.0))
    colsize!(fig.layout, 3, Aspect(1, 1.0))
    colgap!(fig.layout, 10)
    resize_to_layout!(fig)
    return fig
end

function plot_all_vft_ternaries()
    model_id = "mist-conductivity-27.0M-2mpg8dcd"
    model = Mixtures.load_conductivity_model(joinpath(DATA_DIR, "models", model_id)).to("mps")

    for excess in [true, false]
        for parameter in ["Tg", "Ea"]
            for solvent in ["O=C1OCC(F)O1", "CCOC(=O)OC", "CCOC(=O)OCC", "COC(=O)OC"]
                for salt in salts
                    fn_name = "$(parameter)_$(label_smi(solvent))_$(label_smi(salt))"
                    fn_name = excess ? "excess_$(fn_name)" : fn_name
                    mixture = Dict(
                        "solvents" => ["CC1COC(=O)O1", "O=C1OCCO1", solvent],
                        "temperature" => 298.15,
                        "salt" => [salt, "[Li+]"],
                    )
                    with_theme(MISTStyle.theme()) do
                        plot_ternary_vft_parameter(model, mixture; parameter, excess)
                    end |> MISTStyle.savefig(fn_name)
                end
            end
        end
    end
    return
end

function calculate_maximum_relative_excess(model, temperature, solvents)
    dfs = DataFrame[]
    for salt_comp in range(0.02, stop = 0.16, length = 8)
        for salt in salts
            mixture = Dict(
                "solvents" => solvents,
                "temperature" => temperature,
                "salt" => [salt, "[Li+]"],
            )
            df = calculate_excess(model, mixture, salt_comp, 15)
            push!(dfs, df)
        end
    end
    df_all = vcat(dfs...)
    df_all[!, "salt_anion"] = [comp[4] for comp in df_all.components]
    for parameter in ["Tg", "Ea"]
        transform!(
            groupby(df_all, :salt_anion),
            "relative_excess_$(parameter)" => maximum => "maximum_abs_rel_excess_$(parameter)",
        )
    end
    return df_all
end

function plot_soap_similarity_correlation!(f, model, temperature)

    xlabel = "ReMATCH Similarity"
    ax_Ea = Axis(f[1, 1];
        xlabel = xlabel, ylabel = L"max \left| \frac{E_a^{excess}}{E_a} \right|",)
    ax_T0 = Axis(f[1, 2];
        xlabel = xlabel, ylabel = L"max \left| \frac{T_0^{excess}}{T_0} \right|",
    )
    axs = Dict(
        "Ea" => ax_Ea,
        "Tg" => ax_T0,
    )
    all_solvents = [
        "O=C1OCC(F)O1", "CC1COC(=O)O1", "CCOC(=O)OC",
        "CCOC(=O)OCC", "O=C1OCCO1", "COC(=O)OC",
    ]

    solvent_combinations = collect(combinations(all_solvents, 3))
    dfs = DataFrame[]
    for solvents in solvent_combinations
        df = calculate_maximum_relative_excess(model, temperature, solvents)
        solvent_nms = label_smi.(solvents)
        transform!(df,
            [:components, :composition] => ByRow((solv, comp) ->
                weighted_ternary_similarity(collect(solv[1:3]), collect(comp[1:3]))
            ) => :similarity,
        )
        push!(dfs, df)
    end
    df_all = vcat(dfs...)
    unique!(df_all, :components)

    df_all[!, :solvent_label] .= map(row -> join(row[:components][1:3], ", "), eachrow(df_all))
    anion_labels = categorical([comp[4] for comp in df_all.components])

    for (idx, parameter) in enumerate(["Ea", "Tg"])
        h = scatter!(axs[parameter], df_all.similarity, df_all[!, "maximum_abs_rel_excess_$(parameter)"];
            color = levelcode.(anion_labels),
            colormap = MISTStyle.CAT_COLORS,
            colorrange=(1, 10),
            marker=:circle,
            alpha = 0.5,
        )
        points = collect(zip(df_all.similarity, df_all[!, "maximum_abs_rel_excess_$(parameter)"]))

        annotation!(axs[parameter], points, text = df_all.solvent_label)

        if idx == 1
            elements = map(enumerate(levels(anion_labels))) do (i, label)
                MarkerElement(
                    markersize=4pt,
                    marker=h.marker,
                    color=MISTStyle.CAT_COLORS[i],
                    label=label_smi(label)
                )
            end

            Legend(f[1, 1], elements, label.(elements);
                labelsize=5pt,
                tellheight=false,
                tellwidth=false,
                padding=(1, 1, 1, 1),
                margin=(1, 1, 1, 1),
                patchlabelgap=0,
                rowgap=0,
                colgap=0,
                halign=:left,
                valign=:bottom,
                alignmode=Outside(),
            )
        end
    end

    return f
end

function plot_soap_similarity_correlation()
    model_id = "mist-conductivity-27.0M-2mpg8dcd"
    model = Mixtures.load_conductivity_model(joinpath(DATA_DIR, "models", model_id)).to("mps")
    fig = Figure(size = (150mm, 40mm), figure_padding = (2, 2, 2, 2))
    for temperature in [260, 298, 330]
        fn_name = "soap_corr_T_$(temperature)"
        with_theme(MISTStyle.theme()) do
            plot_soap_similarity_correlation!(fig, model, temperature)
        end |> MISTStyle.savefig(fn_name)
    end
    return
end
