using DataFrames
using Makie
using CSV: CSV
using MISTStyle
using Format: format
using GLM
using Interpolations
using Statistics: mean
using Mixtures: Mixtures, titlecase
using Printf

include("TernaryPlots.jl")

const DATA_DIR = realpath(joinpath(pkgdir(Mixtures), "..", "..", "data"))

function label_smi(smi::AbstractString)
    known = Dict(
        "O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F" => "TFSI",
        "F[P-](F)(F)(F)(F)F" => "PF6",
        "O=C1OCC(F)O1" => "FEC",
        "CC1COC(=O)O1" => "PC",
        "CCOC(=O)OC" => "EMC",
        "CCOC(=O)OCC" => "DEC",
        "O=C1OCCO1" => "EC",
        "COC(=O)OC" => "DMC"
    )
    return get(known, smi, smi)
end

function plot_angell_solvents()

    df = DataFrame(CSV.File(joinpath(DATA_DIR, "mixtures", "ionic_conductivity_inference.csv")))

    fig = Figure(size=(53mm, 32mm), figure_padding=(2, 2, 2, 2))

    ax = Axis(fig[1, 1],
        xlabel = L"T_g/T",
        ylabel = L"ln $\sigma$",
        xgridvisible = false,
        ygridvisible = false,
        topspinevisible = true,
        rightspinevisible = true
    )


    # Get unique solvents and create color mapping
    solvents = unique(df[!, "Solvent"])
    n_solvents = length(solvents)
    colors = MISTStyle.CAT_COLORS[1:n_solvents]  # Use Wong colorblind-friendly palette

    # Plot each solvent separately for discrete coloring
    for (i, solvent) in enumerate(solvents)
        mask = df[!, "Solvent"] .== solvent
        scatter!(ax, df[mask, "Tg/T"], df[mask, "conductivity"],
            color = colors[i],
            marker = :circle,
            markersize = 2,
            alpha = 0.6,
            label = string(solvent)
        )
    end

    # Add legend
    Legend(fig[1, 2], ax,
        framevisible = true,
    )

    return fig
end


function calculate_excess(model, mixture::Dict, salt_comp::Float64, n::Integer)

    df = Mixtures.evaluate_conductivity(model, [mixture]; n = n, fixed_salt=salt_comp)

    comp_matrix = reduce(hcat, df.composition)'
    # update salt composition to account for grid discretization
    salt_comp = first(comp_matrix[: , 4])
    pure_solvent_Ea = zeros(Float64, 3)
    pure_solvent_comp = 1 - salt_comp

    for (idx, solvent) in enumerate(mixture["solvents"])
        comp = zeros(4)
        comp[idx] = pure_solvent_comp
        comp[4] = salt_comp
        preds =  Mixtures.evaluate_at_composition(model, mixture, comp)
        pure_solvent_Ea[idx] = preds["Ea"]
    end


    solvent_comps = comp_matrix[:, 1:3]
    solvent_fractions = solvent_comps ./ sum(solvent_comps, dims=2)
    df[!, "ideal_mixing_Ea"] = solvent_fractions * pure_solvent_Ea
    df[!, "excess_Ea"] = df.Ea .- df.ideal_mixing_Ea
    @assert sum(abs.(collect(df.excess_Ea[1:4]))) < 1e-4 "Zero excess for expected degenerate case"
    return df
end

function ternary_activation_energy!(
    ax::Axis, df::DataFrame,
    solvent_names::Vector{String}, global_crange::NTuple{2, Real};
    excess:: Bool = false
    )

    comp_matrix = reduce(hcat, df.composition)'
    a = comp_matrix[:, 1]
    b = comp_matrix[:, 2]
    c = comp_matrix[:, 3]

    if excess
        values = df.excess_Ea
    else
        values = df.Ea
    end
    ternary!(ax, a, b, c, values,
            colormap = MISTStyle.CONTINUOUS_COLORS,
            colorrange = global_crange,
            label_a = label_smi(solvent_names[1]),
            label_b = label_smi(solvent_names[2]),
            label_c = label_smi(solvent_names[3]),
            show_ticks = true)

end

function ternary_activation_energy(model, mixture::Dict; excess::Bool=false, n::Integer = 64)

    fig = Figure(; size=(183mm, 55mm), figure_padding=(1, 1, 1, 1))

    solvent_names = mixture["solvents"]
    mixtures = [mixture]

    all_dfs = []
    for salt_comp in [0.05, 0.1, 0.15]
        if excess
            df = calculate_excess(model, mixture, salt_comp, n )
        else
            df = Mixtures.evaluate_conductivity(model, mixtures; n = n, fixed_salt=salt_comp)
        end
        push!(all_dfs, df)
    end

    if excess
        all_ea = vcat([df.excess_Ea for df in all_dfs]...)
    else
        all_ea = vcat([df.Ea for df in all_dfs]...)
    end

    global_crange = extrema(all_ea)

    for (idx, df) in enumerate(all_dfs)
        ax = Axis(
            fig[1, idx];
            aspect=DataAspect(),
            tellwidth=true,
            tellheight=true,
            limits=((-0.1, 1.1), (-0.2, 1.0)),
        )
        ternary_activation_energy!(ax, df, solvent_names, global_crange; excess = excess)
        salt_comp = first(last.(df.composition))
        hidedecorations!(ax)
        hidespines!(ax)

        val = @sprintf("%.2f", salt_comp)
        label = L"$x_{\mathrm{Li}^{+}}$ = %$val"
        Label(
            fig[0, idx], label;
            tellheight=true,  fontsize=8pt, lineheight=0.1, padding=(0, 0, 0, 0)
        )
    end

    label = excess ? L"excess $E_a$" : L"$E_a$"

    Colorbar(fig[1, 4],
             colormap = MISTStyle.CONTINUOUS_COLORS,
            limits = global_crange,
            label = label,
            height = Relative(0.8),
            tellheight=true,
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


function plot_ternary_activation_energies()
    model_id = "mist-conductivity-27.0M-2mpg8dcd"
    model = Mixtures.load_conductivity_model(joinpath(DATA_DIR, "models", model_id)).to("mps")
    for solvent in ["O=C1OCC(F)O1", "CCOC(=O)OC", "CCOC(=O)OCC", "COC(=O)OC"]
        for salt in ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",  "F[P-](F)(F)(F)(F)F"]
            fn_name = "$(label_smi(solvent))_$(label_smi(salt))"
            mixture = Dict(
                "solvents" => [ "CC1COC(=O)O1", "O=C1OCCO1", solvent],
                "temperature" => 298.15,
                "salt" => [ salt, "[Li+]" ]
            )
            with_theme(MISTStyle.theme()) do
                ternary_activation_energy(model, mixture)
            end |> MISTStyle.savefig(fn_name)
        end
    end
    return
end

function plot_excess_activation_energies()
    model_id = "mist-conductivity-27.0M-2mpg8dcd"
    model = Mixtures.load_conductivity_model(joinpath(DATA_DIR, "models", model_id)).to("mps")
    for solvent in ["O=C1OCC(F)O1", "CCOC(=O)OC", "CCOC(=O)OCC", "COC(=O)OC"]
        for salt in ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F",  "F[P-](F)(F)(F)(F)F"]
            fn_name = "excess_$(label_smi(solvent))_$(label_smi(salt))"
            mixture = Dict(
                "solvents" => [ "CC1COC(=O)O1", "O=C1OCCO1", solvent],
                "temperature" => 298.15,
                "salt" => [salt, "[Li+]"]
            )
            with_theme(MISTStyle.theme()) do
                ternary_activation_energy(model, mixture; excess = true)
            end |> MISTStyle.savefig(fn_name)
        end
    end
    return
end
