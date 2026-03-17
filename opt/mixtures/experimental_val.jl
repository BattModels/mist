using Makie
using MISTStyle
using PythonCall: pyimport, pyconvert
using DataFrames
using CSV: CSV
using StatsBase: mean
using Mixtures


function kekulize(smi::String)
    Chem = pyimport("rdkit.Chem")
    mol = Chem.MolFromSmiles(smi)
    Chem.Kekulize(mol)
    return pyconvert(String, Chem.MolToSmiles(mol; kekuleSmiles=true))
end

function get_smiles(cas_nums::Vector{String})
    cirpy = pyimport("cirpy")
    smiles = String[]
    for cas in cas_nums
        result = cirpy.resolve(cas, "smiles")
        smi = pyconvert(Union{String, Nothing}, result)
        @info cas, smi
        if isnothing(smi)
            push!(smiles, "")
        else
            push!(smiles, kekulize(smi))
        end
    end
    return smiles
end

function smiles_to_iupac(smi::String)
    cirpy = pyimport("cirpy")
    result = cirpy.resolve(smi, "iupac_name")
    name = pyconvert(Union{String, Nothing}, result)
    @info smi, name
    return isnothing(name) ? smi : name
end


function find_sign_flip(molecules::Vector{String}, model_name::String; temperature=298.15, n=10)

    valid_molecules = filter(!isempty, molecules)
    unique!(valid_molecules)
    mixtures = [
        Dict("compounds" => [mol1, mol2], "temperature" => temperature)
        for (i, mol1) in enumerate(valid_molecules)
        for mol2 in valid_molecules[i+1:end]
    ]

    @info length(mixtures)

    df_pred = Mixtures.evaluate_mixtures_hf_ckpt(model_name, mixtures; n=n)

    gdf = groupby(df_pred, :compounds)

    density_flip_pairs = [key.compounds for (key, subdf) in pairs(gdf)
                          if sum(subdf.density_excess .> 0) > 5 && sum(subdf.density_excess .< 0) > 5]
    molar_volume_flip_pairs = [key.compounds for (key, subdf) in pairs(gdf)
                               if sum(subdf.molar_volume_excess .> 0) > 5 && sum(subdf.molar_volume_excess .< 0) > 5]

    df_density_flip = filter(row -> row.compounds in density_flip_pairs, df_pred)
    df_molar_volume_flip = filter(row -> row.compounds in molar_volume_flip_pairs, df_pred)

    return (density_flip=df_density_flip, molar_volume_flip=df_molar_volume_flip)
end


function plot_sign_flip_compounds(cas_nums::Vector{String}, model_name::String; temperature=298.15, n=30)
    molecules = get_smiles(cas_nums[1:2:end])
    data = find_sign_flip(molecules, model_name; temperature, n)

    f = Figure(; size=(120mm, 80mm), figure_padding=(8, 8, 8, 8))

    ax = Axis(f[1, 1];
        xlabel=L"x_1",
        limits =((0., 1.), (nothing, nothing)),
        ylabel=L"$\rho_{rel}^E$",
        ytickformat=values -> ["$(round(v * 100; digits=3))%" for v in values],
        xtickformat="{:.0%}",
    )

    markers = [:circle, :rect, :diamond, :utriangle, :dtriangle, :cross, :xcross, :star5]

    data.density_flip.x1 = first.(data.density_flip.composition)
    handles = []
    labels = String[]
    for (idx, gdf) in enumerate(groupby(data.density_flip, :compounds))
        sort!(gdf, :x1)
        compounds = first(gdf.compounds)
        label = "$(lowercase(smiles_to_iupac(compounds[1]))) & $(lowercase(smiles_to_iupac(compounds[2])))"
        mk = markers[mod1(idx, length(markers))]
        h = scatterlines!(ax, gdf.x1, gdf.density_excess./gdf.density; marker=mk, markersize=3, color=Cycled(idx))
        push!(handles, h)
        push!(labels, label)
    end
    hlines!(ax, [0]; color=:gray, linestyle=:dash, linewidth=0.5)

    if !isempty(handles)
        Legend(f[2, 1], handles, labels;
            halign=:right, valign=:top, orientation=:vertical, nbanks=2)
    end

    return f
end

function plot_sign_flip_compounds()
    df = DataFrame(CSV.File("Viswanathan Lab Inventory.csv"))
    cas_nums = String.(skipmissing(df[!, "CAS #"]))
    with_theme(MISTStyle.theme()) do
        plot_sign_flip_compounds(cas_nums, "mist-models/mist-mixtures-zffffbex")
    end |> MISTStyle.savefig("sign_flip")
    return 0
end
