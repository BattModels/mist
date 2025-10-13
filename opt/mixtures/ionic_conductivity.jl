using DataFrames
using Makie
using JSON
using CSV: CSV
using MISTStyle
using Format: format
using GLM
using Interpolations
using Statistics: mean
using Mixtures: Mixtures
using Printf
using ColorSchemes

const DATA_DIR = realpath(joinpath(pkgdir(Mixtures), "..", "..", "data"))
similarity_data = JSON.parsefile("solvent_rematch.json")

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

function get_similarity(smiles1, smiles2)

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
            total_weight += weight
        end
    end

    return weighted_sum
end

function calculate_excess(model, mixture::Dict, salt_comp::Float64, n::Integer, parameter::String)

    df = Mixtures.evaluate_conductivity(model, [mixture]; n = n, fixed_salt = salt_comp)

    comp_matrix = reduce(hcat, df.composition)'
    # Update salt composition to account for grid discretization
    salt_comp = first(comp_matrix[:, 4])
    pure_solvent_param = zeros(Float64, 3)
    pure_solvent_comp = 1 - salt_comp

    for (idx, solvent) in enumerate(mixture["solvents"])
        comp = zeros(4)
        comp[idx] = pure_solvent_comp
        comp[4] = salt_comp
        preds = Mixtures.evaluate_at_composition(model, mixture, comp)
        pure_solvent_param[idx] = preds[parameter]
    end


    solvent_comps = comp_matrix[:, 1:3]
    solvent_fractions = solvent_comps ./ sum(solvent_comps, dims = 2)
    df[!, "ideal_mixing_$(parameter)"] = solvent_fractions * pure_solvent_Ea
    df[!, "excess_$(parameter)"] = df[!, parameter] .- df[!, "ideal_mixing_$(parameter)"]

    @assert sum(abs.(collect(df.excess_Ea[1:4]))) < 1e-4 "Non-zero excess for expected degenerate case"
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
    salts = ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "F[P-](F)(F)(F)(F)F"]
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
        color = MISTStyle.UM_COLORS.maize, alpha = 0.2, strokewidth = 0
    )

    axislegend(ax, position = :rt, padding = (1, 1, 1, 1), margin = (1, 1, 1, 1), unique = true)
    return f
end

function plot_composition_curves()

    solvents = [
        ["O=C1OCC(F)O1", "CCOC(=O)OC", "O=C1OCCO1"],
        ["CC1COC(=O)O1", "CCOC(=O)OCC", "CCOC(=O)OC"],
        ["CC1COC(=O)O1", "CCOC(=O)OC", "COC(=O)OC"],
        ["CCOC(=O)OCC", "CCOC(=O)OC", "O=C1OCCO1"],
    ]
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
    solvents = [
        ["CC1COC(=O)O1", "O=C1OCCO1", "O=C1OCC(F)O1"],
        ["CC1COC(=O)O1", "CCOC(=O)OCC", "O=C1OCC(F)O1"],
        ["CCOC(=O)OC", "O=C1OCCO1", "O=C1OCC(F)O1"],
        ["CC1COC(=O)O1", "CCOC(=O)OC", "O=C1OCC(F)O1"],
    ]
    salts = ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "F[P-](F)(F)(F)(F)F"]
    temperatures = [260, 280, 300, 320, 340]
    salt_comps = range(start = 0.05, stop = 0.2, length = 5)

    fig = Figure(size = (90mm, 100mm), figure_padding = (2, 2, 2, 2))
    cb = Colorbar(fig[length(solvents) + 1, 1];
        label = L"$$Temperature [K]",
        colormap = MISTStyle.CONTINUOUS_COLORS, colorrange = extrema(temperatures),
        tellheight = true, tellwidth = false, flipaxis = false, vertical = false
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

    solvents = unique(df[!, "Solvent"])
    n_solvents = length(solvents)
    colors = MISTStyle.CAT_COLORS[1:n_solvents]

    for (i, solvent) in enumerate(solvents)
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

function plot_all_vft_ternaries()
    model_id = "mist-conductivity-27.0M-2mpg8dcd"
    model = Mixtures.load_conductivity_model(joinpath(DATA_DIR, "models", model_id)).to("mps")

    for excess in [true, false]
        for parameter in ["Tg", "Ea"]
            for solvent in ["O=C1OCC(F)O1", "CCOC(=O)OC", "CCOC(=O)OCC", "COC(=O)OC"]
                for salt in ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "F[P-](F)(F)(F)(F)F"]
                    fn_name = "$(label_smi(solvent))_$(label_smi(salt))"
                    fn_name = excess ? "excess_$(fn_name)" : fn_name
                    mixture = Dict(
                        "solvents" => ["CC1COC(=O)O1", "O=C1OCCO1", solvent],
                        "temperature" => 298.15,
                        "salt" => [salt, "[Li+]"],
                    )
                    with_theme(MISTStyle.theme()) do
                        plot_ternary_vft_parameter(model, mixture, parameter, excess)
                    end |> MISTStyle.savefig(fn_name)
                end
            end
        end
    end
    return
end

function plot_soap_similarity_correlation!(f, parameter, model, temperature)
    return
end

function plot_soap_similarity_correlation(model, temperature)
    fig = Figure(size = (150mm, 75mm), figure_padding = (2, 2, 2, 2))
    for (idx, parameter) in enumerate(["Tg", "Ea"])
        plot_soap_similarity_correlation!(f[idx], parameter)
    end
    return
end
