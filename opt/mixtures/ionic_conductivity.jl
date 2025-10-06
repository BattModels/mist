using DataFrames
using Makie
using CSV: CSV
using MISTStyle
using Format: format
using GLM
using Interpolations
using Statistics: mean
using Mixtures: Mixtures, titlecase
include("TernaryPlots.jl")

const DATA_DIR = realpath(joinpath(pkgdir(Mixtures), "..", "..", "data"))

function label_smi(smi::AbstractString)
    known = Dict(
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

function plot_solvent_ternary(f, solvent)
    return
end

function ternary_activation_energy!(fig, df::DataFrame)

    # Get solvent names for axis labels
    solvent_names = mixtures[1]["solvents"]

    # Determine global colorrange for consistent coloring
    global_crange = extrema(df.Ea)
    println(global_crange)
    @info "Activation Energy Range" global_crange

    ax = Axis(fig[1, 1], aspect=DataAspect())

    # Extract first 3 components (solvents) from composition
    comp_matrix = reduce(hcat, df.composition)'  # Convert to matrix
    a = comp_matrix[:, 1]
    b = comp_matrix[:, 2]
    c = comp_matrix[:, 3]

    ternary!(ax, a, b, c, df.Ea,
            colormap = :viridis,
            colorrange = global_crange,
            label_a = solvent_names[1],
            label_b = solvent_names[2],
            label_c = solvent_names[3],
            show_ticks = true)

    hidedecorations!(ax)
    hidespines!(ax)

    # Add title showing salt composition
    salt_comp = first(last.(df.composition))
    Label(fig[0, 1], "Salt: $(round(salt_comp, digits=3))", fontsize=14)

    return fig
end

function ternary_activation_energy(model_id::String)
    model = Mixtures.load_conductivity_model(joinpath(DATA_DIR, "models", model_id)).to("mps")

    fig = Figure(size=(300*3, 400))

    mixtures = [
        Dict(
            "solvents" => ["O=C1OCC(F)O1", "CCOC(=O)OC", "O=C1OCCO1"],
            "temperature" => 298.15,
            "salt" => ["O=S(=O)([N-]S(=O)(=O)C(F)(F)F)C(F)(F)F", "[Li+]"]
        ),
    ]

    for (idx, salt_comp) in enumerate([0.06, 0.1, 0.14])
        df = Mixtures.evaluate_conductivity(model, mixtures; n = 140, fixed_salt=salt_comp)
        ternary_activation_energy!(fig[:, idx], df)
    end
    # Add colorbar
    Colorbar(
        fig[:, 4],
        tellheight = true,
        tellwidth = true,
        colormap = :viridis,
        limits = (100, 200),
        label = L"$E_a$"
    )
    return  fig
end
