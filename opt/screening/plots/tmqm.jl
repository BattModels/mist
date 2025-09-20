using ScreeningPlots
using ScreeningPlots: HARTREE_TO_EV
using Makie
using MISTStyle
using PythonCall
using DataFrames
using StatsBase: StatsBase, cor
using CategoricalArrays: categorical, levelcode

ROOTDIR = strip(read(`git rev-parse --show-toplevel`, String))
DATA_DIR = joinpath(ROOTDIR, "data")

function tmqm_dataset(path, split="validation")
    ds = pyimport("datasets").load_dataset(path)
    df = DataFrame(;
        smiles=pyconvert(Vector{String}, ds[split]["smiles"]),
        cid=pyconvert(Vector{String}, ds[split]["CSD_code"]),
        homo=pyconvert(Vector{Float64}, ds[split]["HOMO_Energy"]),
        lumo=pyconvert(Vector{Float64}, ds[split]["LUMO_Energy"]),
        gap=pyconvert(Vector{Float64}, ds[split]["HL_Gap"]),
        electronic_E=pyconvert(Vector{Float64}, ds[split]["Electronic_E"]),
        dispersion_E=pyconvert(Vector{Float64}, ds[split]["Dispersion_E"]),
        dipole_moment=pyconvert(Vector{Float64}, ds[split]["Dipole_M"]),
        metal_q=pyconvert(Vector{Float64}, ds[split]["Metal_q"]),
        polarizability=pyconvert(Vector{Float64}, ds[split]["Polarizability"]),
    )
    unique!(df, [:smiles, :cid])
    return df
end

function predict_dataset(model::Py, dataset::Py)
    smi = pyconvert(Vector{String}, dataset["smiles"])
    return ScreeningPlots.predict_mist(model, smi; encoding="smiles")
end

"""
    find_chiral_pairs(df_cid::DataFrame)

Identify molecules which collapse to the same encoding where stereochemistry is omitted.
Filtering to molecules with at least 2 different encodings
"""
function find_chiral_pairs(df_cid)
    df = transform(df_cid,
        :smiles => ByRow(ScreeningPlots.isomeric_smiles) => :isomeric_smiles,
    )
    n_pairs = combine(groupby(df, :isomeric_smiles), :smiles => length∘unique)
    subset!(n_pairs, :smiles_length_unique => ByRow(>=(2)))
    subset!(df, :isomeric_smiles => ByRow(∈(n_pairs.isomeric_smiles)))
    return df
end

function sterochemistry_examples(df_tmqm)
    stero_pairs = find_chiral_pairs(df_tmqm)
end

function compare_stero(mist, df_stero)
    df_mist = ScreeningPlots.predict_mist(mist, df_stero.smiles; encoding="smiles")
    rename!(df_mist,
        "HOMO_Energy" => "homo",
        "LUMO_Energy" => "lumo",
        "HL_Gap" => "gap",
        "Electronic_E" => "electronic_E",
        "Dispersion_E" => "dispersion_E",
        "Dipole_M" => "dipole_moment",
        "Metal_q" => "metal_q",
        "Polarizability" => "polarizability",
    )
    df_mist = leftjoin(df_mist, df_stero;
        on=:smiles,
        makeunique=true,
        renamecols="_mist" => "_ref",
    )
    rename!(df_mist, "isomeric_smiles_ref" => "isomeric_smiles", "cid_ref" => "cid")

    return df_mist
end

function figure_stero(df_mist)
    f = Figure(; size=(160, 95))
    sort!(df_mist, :isomeric_smiles)
    df_mist = dropmissing(df_mist)
    df_mist.isomeric_smiles = categorical(df_mist.isomeric_smiles)

    ax_args = (;
        xticks=WilkinsonTicks(3),
        yticks=WilkinsonTicks(3),
    )
    sargs = (;
        strokewidth=0.5pt,
        strokecolor=:black,
        color=levelcode.(df_mist.isomeric_smiles),
        colormap=:glasbey_hv_n256,
        marker=:circle,
        markersize=4pt,
    )
    abargs = (; color=:black, linestyle=:dash)

    known_cid = [
        "LEYLAC",
        "LIMTIK",
        "ROMVAQ",
        "ROMVEU",
    ]

    function parity!(f, df, chn; title=nothing)
        ax = Axis(f; title, ax_args...)
        ablines!(ax, 0, 1; abargs...)
        x = Symbol(chn, "_ref")
        y = Symbol(chn, "_mist")
        scatter!(ax, df[!, x], df[!, y]; sargs...)
        df_known = subset(df, :cid => ByRow(∈(known_cid)))
        annotation!(ax, collect(zip(df_known[!, x], df_known[!, y]));
            text=df_known.cid,
            fontsize=4pt,
            linewidth=0.5pt,
            shrink=(1pt, 2pt),
            lineheight=0.
        )
        @info title cor(df[!, x], df[!, y])
    end

    parity!(f[1, 1], df_mist, :metal_q; title="Metal Natural Charge (e)")
    df_mist.gap_mist .*= HARTREE_TO_EV
    df_mist.gap_ref .*= HARTREE_TO_EV
    parity!(f[1, 2], df_mist, :gap; title="Gap (eV)")

    # parity!(f[2, 1], df_mist, :homo_ref, :homo_mist; title="HOMO")
    # parity!(f[2, 2], df_mist, :gap_ref, :gap_mist; title="Gap")

    # parity!(f[1, 3], df_mist, :dipole_moment_ref, :dipole_moment_mist; title="Dipole Moment")
    # parity!(f[2, 3], df_mist, :polarizability_ref, :polarizability_mist; title="Polarizability")

    Label(f[:, 0], "MIST Predictions";
        rotation=pi/2,
        tellwidth=true,
        tellheight=false
    )
    Label(f[end+1, 1:end], "Reference";
        tellwidth=false,
        tellheight=true
    )


    return f
end
