using Makie
using MISTStyle
using DataFrames
using CSV: CSV
using Interpolations: linear_interpolation
using StatsBase: mad


function load_data_yang(mist_file, yang_file)
    # Load Yank Measurements
    df_yang = DataFrame(CSV.File(yang_file))
    rename!(df_yang,
        "T [K]" => "temperature",
        "rho [g/cm-3]" => "density",
        "η [mPa·s]" => "viscosity",
        "VE [cm3 mol-1]" => "excess_molar_volume",
    )
    # Invalid excess molar volume for Heptane at 333.15K
    filter!(row -> !(row.temperature in [333.15, 343.15] && row.Compound2 == "Heptane") , df_yang)

    # Replace Compounds with SMILES
    smi_lookup = Dict(
        "DMC" => "COC(=O)OC",
        "Chlorobenzene" => "Clc1ccccc1",
        "Heptane" => "CCCCCCC",
        "Hexane" => "CCCCCC",
    )
    transform!(df_yang,
        :Compound1 => ByRow(x -> smi_lookup[x]) => :smi1,
        :Compound2 => ByRow(x -> smi_lookup[x]) => :smi2,
    )
    select!(df_yang, Not([:Compound1, :Compound2]))


    # Load MIST Predictions
    df_mist = DataFrame(CSV.File(mist_file))
    rename!(df_mist, "predicted" => "excess_molar_volume")
    select!(df_mist, Not([1, 2]))
    select!(df_mist, Not("excess_molar_volume/(cm3/mol)"))
    return df_yang, df_mist
end

function load_data(inference, ref)
    df_inf = DataFrame(CSV.File(inference))
    df_ref = DataFrame(CSV.File(ref))
    select!(df_inf, [:x1, :smi1, :smi2, :temperature, Symbol("excess_molar_volume/(cm3/mol)")])
    rename!(df_inf, "excess_molar_volume/(cm3/mol)" => "excess_molar_volume")
    select!(df_ref, [:x1, :smi1, :smi2, :temperature, Symbol("excess_molar_volume/(cm3/mol)")])
    rename!(df_ref, "excess_molar_volume/(cm3/mol)" => "excess_molar_volume")
    return df_inf, df_ref
end

figure_parity(args...; kwargs...) = figure_parity!(Figure(), args...)

function figure_parity!(f, df_x, df_y)
    cols = [:temperature, :x1, :smi1, :smi2, :excess_molar_volume]
    df = select(df_x, cols)
    df_y = select(df_y, cols)
    itp_y = linear_interpolation(df_y.x1, df_y.excess_molar_volume)
    df.excess_molar_volume_y = itp_y(df.x1)
end

function figure_excess_molar_volume(df_model, df_ref)
    f = Figure()
    gl = GridLayout(f[1, 1])
    cb = Colorbar(f[2, 1];
        label = L"Temperature [K]$$",
        colormap = MISTStyle.CONTINUOUS_COLORS,
        colorrange = (290, 350),
        vertical = false,
        tellheight = true,
        tellwidth = false,
        flipaxis = false,
    )
    # plot_excess_molar_volume!(gl[1, 1], gl[1,2], cb, df_model, df_ref; smi2="COCCOC", xvisible=false)
    # plot_excess_molar_volume!(gl[2, 1], gl[2, 2], cb, df_model, df_ref; smi2="C1COCO1")
    plot_excess_molar_volume!(gl[1, 1], gl[1,2], cb, df_model, df_ref; smi2="CCCCCC", xvisible=false)
    plot_excess_molar_volume!(gl[2, 1], gl[2, 2], cb, df_model, df_ref; smi2="CCCCCCC")

    return f
end

function plot_excess_molar_volume!(f1, f2, cb::Colorbar, df_model, df_ref; smi1="COC(=O)OC", smi2="CCCCCCC", xvisible=true)
    ax = Axis(f1;
        xlabel = L"x_1",
        ylabel = L"$V_m$ [cm$^3$/mol]",
        limits = ((0, 1), (0, nothing)),
        xlabelvisible = xvisible,
        xticksvisible = xvisible,
        xticklabelsvisible = xvisible,
    )
    df_model = subset(df_model, :smi1 => ByRow(==(smi1)), :smi2 => ByRow(==(smi2)))
    df_ref = subset(df_ref, :smi1 => ByRow(==(smi1)), :smi2 => ByRow(==(smi2)))
    for gdf in groupby(df_model, :temperature)
        line = lines!(ax, gdf.x1, gdf.excess_molar_volume;
            color=gdf.temperature,
            MISTStyle.cb_attrs(cb, Lines)...
        )
    end
    for gdf in groupby(df_ref, :temperature)
        scatter!(ax, gdf.x1, gdf.excess_molar_volume;
            color=gdf.temperature,
            MISTStyle.cb_attrs(cb, Lines)...
        )
    end

    df = combine(groupby(df_ref, [:temperature, :smi1, :smi2])) do gdf
        gdf_model = subset(df_model,
            :smi1 => ByRow(==(first(gdf.smi1))),
            :smi2 => ByRow(==(first(gdf.smi2))),
            :temperature => ByRow(==(first(gdf.temperature))),
        )
        sort!(gdf_model, :x1)
        itp = linear_interpolation(gdf_model.x1, gdf_model.excess_molar_volume)
        return (;
            excess_molar_volume_x = gdf.excess_molar_volume,
            excess_molar_volume_y = itp(gdf.x1)
        )

    end

    ax = Axis(f2;
        xlabel = L"Experimental [cm$^3$/mol]",
        ylabel = L"MIST [cm$^3$/mol]",
        xlabelvisible = xvisible,
        xticksvisible = xvisible,
        xticklabelsvisible = xvisible,
    )
    scatter!(ax, df.excess_molar_volume_x, df.excess_molar_volume_y;
        color=df.temperature,
        MISTStyle.cb_attrs(cb, Lines)...
    )

    # Report Accuracy Statistics
    error = df.excess_molar_volume_y .- df.excess_molar_volume_x
    @info "$smi1, $smi2" mae=mad(error) cor(df.excess_molar_volume_y, df.excess_molar_volume_x) rmsd(df.excess_molar_volume_x, df.excess_molar_volume_y)
    ablines!(ax, 0, 1; color=:black, linestyle=:dash)

    return f

end

function plot_alkane_trends(df; temperature=293.15)
    f = Figure()
    ax = Axis(f[1, 1])
    df = subset(df, :temperature => ByRow(==(temperature)))
    df.n_carbon = length.(df.smi2)
    sort!(df, :x1)

    cb = Colorbar(f[1, 2];
        label = L"N Carbon",
        colormap = :managua10,
        colorrange = extrema(df.n_carbon),
        # vertical = false,
        # tellheight = true,
        # tellwidth = false,
        # flipaxis = false,
    )


    for gdf in groupby(df, :n_carbon)
        @info nrow(gdf)
        lines!(ax, gdf.x1, gdf.excess_molar_volume;
            color=gdf.n_carbon,
            MISTStyle.cb_attrs(cb, Lines)...
        )
    end

    return f
end

