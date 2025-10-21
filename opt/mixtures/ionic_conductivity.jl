using DataFrames
using Makie
using JSON
using CSV: CSV
using GLM
using MISTStyle
using MISTStyle: MISTStyle, label
using Format: format
using Statistics: mean
using Mixtures: Mixtures
using Printf
using ColorSchemes
using Combinatorics
using CategoricalArrays: categorical, levelcode

const DATA_DIR = realpath(joinpath(pkgdir(Mixtures), "..", "..", "data"))
const similarity_data = JSON.parsefile(joinpath(DATA_DIR, "mixtures", "solvent_rematch.json"))
const solvents = [
    ["CC1COC(=O)O1", "O=C1OCCO1", "O=C1OCC(F)O1"],
    ["CC1COC(=O)O1", "CCOC(=O)OCC", "O=C1OCC(F)O1"],
    ["CCOC(=O)OC", "O=C1OCCO1", "O=C1OCC(F)O1"],
    ["CC1COC(=O)O1", "CCOC(=O)OC", "O=C1OCC(F)O1"],
]
const salts = ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "F[P-](F)(F)(F)(F)F"]


function label_smi(smi::AbstractString; input::AbstractString = "smi")
    smiles_to_name = Dict(
        "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F" => "TFSI",
        "F[P-](F)(F)(F)(F)F" => "PF6",
        "O=C1OCC(F)O1" => "FEC",
        "CC1COC(=O)O1" => "PC",
        "CCOC(=O)OC" => "EMC",
        "CCOC(=O)OCC" => "DEC",
        "O=C1OCCO1" => "EC",
        "COC(=O)OC" => "DMC",
    )
    name_to_smiles = Dict(v => k for (k, v) in smiles_to_name)
    if input == "smi"
        return get(smiles_to_name, smi, smi)
    else
        return get(name_to_smiles, smi, smi)
    end
end

function get_acronym(smi::AbstractString)
    known = Dict(
        "Fluoroethylene Carbonate" => "FEC",
        "Propylene Carbonate" => "PC",
        "Ethyl Methyl Carbonate" => "EMC",
        "Diethyl Carbonate" => "DEC",
        "Ethylene Carbonate" => "EC",
        "Dimethyl Carbonate" => "DMC",
    )
    return get(known, smi, smi)
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
        composition, conductivity = Mixtures.conductivity_composition_curve(model, mixture)
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

    fig = Figure(size = (90mm, 120mm), figure_padding = (5, 5, 5, 5))
    cb = Colorbar(fig[length(solvents)+1, 1];
        label = L"$$Temperature [K]",
        colormap = MISTStyle.CONTINUOUS_COLORS, colorrange = extrema(temperatures),
        tellheight = true, tellwidth = false, flipaxis = false, vertical = false,
    )

    for (idx, solvent) in enumerate(solvents)
        ax = Axis(
            fig[idx, 1], limits = (nothing, (0.0, nothing)),
            title = "$(label_smi(solvent[1])) | $(label_smi(solvent[2])) |  $(label_smi(solvent[3]))",
            xlabel = idx == length(solvents) ? L"$x_{Li^+}$" : "",
            ylabel = L"$\frac{E_{a,LiPF_6}}{T} - \frac{E_{a,LiTFSI}}{T}$",
            xticksvisible = idx==length(solvents),
            xticklabelsvisible = idx==length(solvents),
            titlesize = 6pt,
            xgridvisible = false,
            ygridvisible = true,
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
        df = Mixtures.evaluate_conductivity(model, mixtures; n = n, fixed_salt = salt_comp)
        if excess
            Mixtures.calculate_excess!(df)
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
            df = Mixtures.evaluate_conductivity(model, [mixture]; n = 15, fixed_salt = salt_comp)
            Mixtures.calculate_excess!(df)
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

    xlabel = "REMatch Similarity"
    ax_Ea = Axis(f[1, 1]; limits = (nothing, (0.0, 0.18)),
        xlabel = xlabel, ylabel = L"max \left| \frac{E_a^{excess}}{E_a} \right|")
    ax_T0 = Axis(f[2, 1]; limits = (nothing, (0.0, 0.18)),
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
                Mixtures.weighted_ternary_similarity(collect(solv[1:3]), collect(comp[1:3]), similarity_data)
            ) => :similarity,
        )
        push!(dfs, df)
    end
    df_all = vcat(dfs...)
    unique!(df_all, :components)

    df_all[!, :solvent_label] .= map(row -> join(label_smi.(row[:components][1:3]), ", "), eachrow(df_all))
    anion_labels = categorical([comp[4] for comp in df_all.components])
    single_salt = unique(df_all, :solvent_label)

    for (idx, parameter) in enumerate(["Ea", "Tg"])
        h = scatter!(
            axs[parameter], df_all.similarity, df_all[!, "maximum_abs_rel_excess_$(parameter)"];
            color = levelcode.(anion_labels),
            colormap = MISTStyle.CAT_COLORS,
            colorrange = (1, 10),
            marker = :circle,
            alpha = 0.6,
        )

        points = collect(
            zip(single_salt[idx:2:end, :similarity], single_salt[idx:2:end, "maximum_abs_rel_excess_$(parameter)"]),
        )
        annotation!(axs[parameter], points; text = single_salt[idx:2:end, :solvent_label], shrink = (0.0, 0.0))

        if idx == 1
            elements = map(enumerate(levels(anion_labels))) do (i, label)
                MarkerElement(
                    markersize = 4pt,
                    marker = h.marker,
                    color = MISTStyle.CAT_COLORS[i],
                    label = label_smi(label),
                )
            end

            Legend(f[1, 1], elements, label.(elements);
                labelsize = 5pt,
                tellheight = false,
                tellwidth = false,
                padding = (1, 1, 1, 1),
                margin = (1, 1, 1, 1),
                patchlabelgap = 0,
                rowgap = 0,
                colgap = 0,
                halign = :left,
                valign = :bottom,
                alignmode = Outside(),
            )
        end
    end

    return f
end


function plot_soap_similarity_correlation()
    model_id = "mist-conductivity-27.0M-2mpg8dcd"
    model = Mixtures.load_conductivity_model(joinpath(DATA_DIR, "models", model_id)).to("mps")
    fig = Figure(size = (90mm, 120mm), figure_padding = (2, 2, 2, 2))
    for temperature in [298] # 298, 330]
        fn_name = "soap_corr_T_$(temperature)"
        with_theme(MISTStyle.theme(); fontsize = 1pt) do
            plot_soap_similarity_correlation!(fig, model, temperature)
        end |> MISTStyle.savefig(fn_name)
    end
    return
end

function plot_non_arr_data!(fig, df)

    ax = Axis(
        fig[1, 1];
        xlabel = L"$\frac{1000}{T}$ [$K^{-1}$]", ylabel = L"$\ln\;\sigma$ [mS/cm]")

    count = 0
    seen_combinations = Set()

    for gdf in groupby(df, [
        "sub1_name", "x1 (mole fraction)" , "sub2_name" , "x2 (mole fraction)" ,
        "sub3_name", "x3 (mole fraction)", "salt", "x4 (anion, mole fraction)"
        ])

        comps = (
            sub1=first(gdf.sub1_name), sub2=first(gdf.sub2_name),
            sub3=first(gdf.sub3_name), salt=first(gdf.salt)
            )

        if comps in seen_combinations
            continue
        end

        if nrow(gdf) > 5
            count += 1
            push!(seen_combinations, comps)
            label = "$(get_acronym(comps.sub1)), $(get_acronym(comps.sub2)), $(get_acronym(comps.sub3)), $(comps.salt)"
            lines!(ax, 1000 ./gdf[!, "temperature (K)"], log.(gdf[!, "ionic conductivity (mS/cm)"]);
                label = label,
            )
            if count > 14
                break
            end
        end

    end
    Legend(fig[2, 1], ax;
            nbanks = 3,
            labelsize = 5pt,
            tellheight = true,
            tellwidth = true,
            padding = (1, 1, 1, 1),
            margin = (1, 1, 1, 1),
            patchlabelgap = 1pt,
            rowgap = 0,
            colgap = 5pt,
            halign = :left,
            valign = :bottom,
            alignmode = Outside(),
    )
    rowgap!(fig.layout, 2)
    return fig
end

function plot_non_arr_data()
    fig = Figure(size = (90mm, 80mm), figure_padding = (2, 2, 2, 2))
    df = DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "published_AEM_data.csv")))
    subset!(df, "x5 (cation, mole fraction)" => ByRow(>(0.14)))
    fn_name = "non_arr_data"
    with_theme(MISTStyle.theme()) do
        plot_non_arr_data!(fig, df)
    end |> MISTStyle.savefig(fn_name)

    return fig
end

function pure_solvent_T0(model, solvent, salt, salt_comp; temperature = 298.15)
    mixture = Dict(
        "solvents" => [solvent,  "CC1COC(=O)O1", "O=C1OCCO1"],
        "temperature" => temperature,
        "salt" => [ salt, "[Li+]" ]
    )
    comp = zeros(4)
    comp[1] = 1 - salt_comp
    comp[4] = salt_comp
    preds =  Mixtures.evaluate_at_composition(model, mixture, comp)
    return preds["Tg"]
end

function plot_T0_vs_melting_point!(fig, model; molarity = 1)
    ax = Axis(
        fig[1, 1];
        limits = ((200, 320), nothing),
        xlabel = L"$$Melting Point [K]", ylabel = L"$T_0$ [K]")

    for solvent in ["O=C1OCC(F)O1", "CC1COC(=O)O1", "CCOC(=O)OC", "O=C1OCCO1", "CCOC(=O)OCC", "COC(=O)OC"]
        solvent_acr = label_smi(solvent)
        for (idx, salt) in enumerate(salts)

            properties = Mixtures.SOLVENT_DATA[solvent_acr]
            salt_comp = Mixtures.molarity_to_mole_fraction(solvent_acr; molarity = molarity)
            @info "Solvent $(solvent_acr) | Molarity $(molarity) | $(round(salt_comp, digits=3))"
            T0 = pure_solvent_T0(model, solvent, salt, salt_comp; temperature = 298.15)
            scatter!(
                ax, properties.mp_C + 273.15, T0;
                color = MISTStyle.CAT_COLORS[idx],
                label = label_smi(salt),
                marker = :circle
            )
            align = salt == salts[1] ? (:right, :top) : (:right, :bottom)
            text!(
                ax, properties.mp_C + 273.15, T0;
                text=solvent_acr,
                align=align,
                offset= (-2, 0.5),
                fontsize = 4pt
            )
        end
    end
    Legend(fig[1, 1], ax;
        unique = true, framevisible = true,
        tellheight = false,
        tellwidth = false,
        padding = (1, 1, 1, 1),
        margin = (1, 1, 1, 1),
        halign = :right,
        valign = :center,
        alignmode = Inside(),
    )
    return fig
end

function plot_T0_vs_melting_point(molarity = 1)
    model_id = "mist-conductivity-27.0M-2mpg8dcd"
    model = Mixtures.load_conductivity_model(joinpath(DATA_DIR, "models", model_id)).to("mps")
    fig = Figure(size = (55mm, 60mm), figure_padding = (5, 5, 5, 5))
    fn_name = "melting_vs_T0_$(molarity)M"
    with_theme(MISTStyle.theme()) do
        plot_T0_vs_melting_point!(fig, model; molarity = molarity)
    end |> MISTStyle.savefig(fn_name)
    return
end

function fit_T0_dielectric_model(model, salt, molarity; temperature = 298.15)
    df = calculate_T0_dataframe(model, salt, molarity; temperature = temperature)
    lm_model = lm(@formula(T0 ~ dielectric + melting_point), df)

    display(lm_model)
    @info "Model Summary" salt=salt R²=round(r2(lm_model), digits=4) adj_R²=round(adjr2(lm_model), digits=4)

    return lm_model, df
end


function plot_T0_model_fit!(fig, lm_model, df, salt)
    df.T0_pred = predict(lm_model, df)
    salt_label = "Li$(label_smi(salt))"
    showy = salt_label=="LiTFSI"
    ax = Axis(fig[1, 1];
            xlabel = L"MIST $T_0$ [K]",
            ylabel = showy ? L"$T_0$ = f($\epsilon$, $T_m$) [K]" : "",
            title = salt_label,
            titlesize = 5pt,
            yticksvisible = showy, yticklabelsvisible = showy
    )

    scatter!(ax, df.T0, df.T0_pred; marker = :circle)
    text!(ax, df.T0, df.T0_pred;
        text = df.solvent, fontsize = 4pt, align = (:left, :center), offset = (2, 0))

    lims = collect(extrema(vcat(df.T0, df.T0_pred)) .+ (-2, 2))
    lines!(ax, lims, lims; color = :orange, linewidth = 0.7)

    stats_text = @sprintf("R² = %.3f\nRMSE = %.2f K", r2(lm_model), sqrt(mean((df.T0 .- df.T0_pred).^2)))
    text!(ax, lims[1] + 2, lims[2] - 2;
          text = stats_text, fontsize = 4pt, align = (:left, :top)
    )

    return fig
end

function compare_T0_models(model, salt, molarity; temperature = 298.15)
    df = calculate_T0_dataframe(model, salt, molarity; temperature = temperature)
    lms = [
        lm(@formula(T0 ~ dielectric), df), lm(@formula(T0 ~ melting_point), df),
        lm(@formula(T0 ~ dielectric + melting_point), df), lm(@formula(T0 ~ dielectric * melting_point), df)
    ]
    for lm_model in lms
        display(lm_model)
        @info "Salt" salt "| \nR² = $(round(r2(lm_model), digits=4)) | Adjusted R² = $(round(adjr2(lm_model), digits=4))"
    end
    return lms, df
end

function calculate_T0_dataframe(model, salt, molarity; temperature = 298.15)
    data = []
    for (solvent_name, props) in Mixtures.SOLVENT_DATA
        solvent_smi = label_smi(solvent_name; input = "acr")
        salt_comp = Mixtures.molarity_to_mole_fraction(solvent_name; molarity = molarity)
        T0 = pure_solvent_T0(model, solvent_smi, salt, salt_comp; temperature = temperature)
        push!(data, (
            solvent = solvent_name,
            solvent_smiles = solvent_smi,
            dielectric = props.dielectric,
            melting_point = props.mp_C + 273.15,
            T0 = T0
        ))
    end
    return DataFrame(data)
end

function fit_and_plot_glm!(fig, model; molarity = 1)
    for (i, salt) in enumerate(salts)
        lm_model, df = fit_T0_dielectric_model(model, salt, molarity)
        plot_T0_model_fit!(fig[1, i], lm_model, df, salt)
    end
    colgap!(fig.layout, 9)
    rowgap!(fig.layout, 0)
    return fig
end

function fit_and_plot_glm(; molarity = 1)
    model_id = "mist-conductivity-27.0M-2mpg8dcd"
    model = Mixtures.load_conductivity_model(joinpath(DATA_DIR, "models", model_id)).to("mps")

    fig = Figure(size = (100mm, 50mm))

    fn_name = "gml_fit_$(molarity)M"
    with_theme(MISTStyle.theme()) do
        fit_and_plot_glm!(fig, model; molarity = molarity)
    end |> MISTStyle.savefig(fn_name)
    return
end
