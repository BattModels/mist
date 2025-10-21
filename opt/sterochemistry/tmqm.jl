using Makie
using MISTStyle
using PythonCall: pyimport
using DataFrames: DataFrame, ByRow, transform, combine, groupby, subset!
using StatsBase: StatsBase, cor
using CategoricalArrays: categorical, levelcode
using JSON: JSON

ROOTDIR = strip(read(`git rev-parse --show-toplevel`, String))
DATA_DIR = joinpath(ROOTDIR, "data")
HARTREE_TO_EV = 27.211_386_245_981

function load_jsonl(file)
    rows = []
    open(file, "r") do io
        for line in eachline(io)
            push!(rows, JSON.parse(line))
        end
    end
    return DataFrame(rows)
end

function isomeric_smiles(smi::String)
    rdkit_chem = pyimport("rdkit.Chem")
    mol = rdkit_chem.MolFromSmiles(smi; sanitize=false)
    isnothing(mol) && return smi
    for atom in mol.GetAtoms()
        atom.SetChiralTag(rdkit_chem.ChiralType.CHI_UNSPECIFIED)
    end
    return pyconvert(String, rdkit_chem.MolToSmiles(mol, isomericSmiles=true))
end

function zero_based_limits(x)
    l, u = extrema(x)
    if 0 <= l && 0 <= u
        return (0.0, nothing)
    elseif l <= 0 && u <= 0
        return (floor(l), 0.0)
    else
        return nothing
    end
end


"""
    find_chiral_pairs(df_cid::DataFrame)

Identify molecules which collapse to the same encoding where stereochemistry is omitted.
Filtering to molecules with at least 2 different encodings
"""
function find_chiral_pairs(df)
    df = transform(df, :smiles => ByRow(label_metal) => [:metal, :stero, :stero_id])
    subset!(df, :metal => ByRow(!isnothing), :stero => ByRow(!=("H")))
    df = transform(df,
        :smiles => ByRow(isomeric_smiles) => :isomeric_smiles,
    )
    n_pairs = combine(groupby(df, :isomeric_smiles), :smiles => length∘unique)
    subset!(n_pairs, :smiles_length_unique => ByRow(>=(2)))
    subset!(df, :isomeric_smiles => ByRow(∈(n_pairs.isomeric_smiles)))
    return df
end

function label_metal(smi)
    m = match(r"\[([A-Za-z]{2})@([A-Za-z]{1,2})(\d{0,2})\]", smi)
    if isnothing(m)
        return (; metal=nothing, stero=nothing, stero_id=nothing)
    else
        return (; metal=m[1], stero=m[2], stero_id=m[3])
    end
end

function figure_chn(df; chn=:HL_Gap)
    df = transform(df,
        :smiles => ByRow(label_metal) => [:metal, :stero, :stero_id]
    )
    subset!(df, :metal => ByRow(!isnothing), :stero => ByRow(!=("H")))
    df.stero = categorical(df.stero)
    df.metal = categorical(df.metal)
    df_stero = find_chiral_pairs(df)
    df_stero.stero = categorical(df_stero.stero)

    stero_range = combine(groupby(df_stero, :isomeric_smiles)) do gdf
        l, u = extrema(gdf[!, chn])
        return u - l
    end
    sort!(stero_range, :x1)
    @info stero_range
    isomers = [
        "CC12C[NH2][Re]([C]O)([C]O)([C]O)([NH2]C1)[NH2]C2",
        "O[N](O)[Co]123([N](O)O)[NH2]CC[NH]1CC[NH]2CC[NH2]3",
        "CC[P](CC)(CC)[Pt]([Cl])([Cl])[P](OC1C(F)[CH][CH][CH]C1F)(OC1C(F)[CH][CH][CH]C1F)C1[CH][CH][CH][CH][CH]1",
    ]
    subset!(df_stero, :isomeric_smiles => ByRow(∈(isomers)))
    df_stero.bar_labels = map(eachrow(df_stero)) do row
        (; metal, stero, stero_id) = row
        return "@$(stero)$(stero_id)"
    end

    f = Figure(; size=(160, 110))
    ax = Axis(f[2, 1];
        ylabel = L"$$Gap (eV)",
        xlabel = "TM Complex",
        limits=(nothing, zero_based_limits(df_stero[!, chn])),
    )

    df_stero = combine(groupby(df_stero, :isomeric_smiles)) do gdf
        gdf.stero_idx = 1:nrow(gdf)
        return gdf
    end
    dropmissing!(df_stero)
    subset!(df_stero, :stero_idx => ByRow(<=(2)))
    sort!(df_stero, chn)
    df_stero.isomeric_smiles = categorical(
        df_stero.isomeric_smiles;
        levels=unique(df_stero.isomeric_smiles)
    )

    @info df_stero[!, [:isomeric_smiles, :smiles, :metal, :stero, :stero_id, :id]]

    barplot!(ax,
        levelcode.(df_stero.isomeric_smiles),
        df_stero[!, Symbol(chn, "_mist")] .* HARTREE_TO_EV;
        bar_labels=df_stero.bar_labels,
        label_rotation=pi/2,
        label_position=:center,
        label_offset=0,
        label_color=:white,
        label_size=6pt,
        dodge=df_stero.stero_idx,
        color=levelcode.(df_stero.isomeric_smiles),
        colormap=MISTStyle.CAT_COLORS,
        colorrange=(1, 10),
    )

    # ax2 = Axis(f[1, 2];
    #     limits = lift(x -> (last(x), last(x)), ax.limits),
    # )
    # scatter!(ax2, df_stero[!, chn], df_stero[!, Symbol(chn, "_mist")])
    # linkyaxes!(ax, ax2)

    rowsize!(f.layout, 2, 40pt)
    resize_to_layout!(f)

    return f
end

function figure_complexes(df, df_stero)
    # transform!(df, :stero => ByRow(x -> uppercasefirst(x)) => :stero)

    df = transform(df,
        :smiles => ByRow(label_metal) => [:metal, :stero, :stero_id]
    )
    subset!(df, :metal => ByRow(!isnothing), :stero => ByRow(!=("H")))
    df.stero = categorical(df.stero)
    df.metal = categorical(df.metal)

    df = unique(df, :id)
    df_stero = find_chiral_pairs(df)
    df_stero.metal = categorical(df_stero.metal)
    df_stero.stero = categorical(df_stero.stero)

    xcol = :HL_Gap
    ycol = :Polarizability
    x = df_stero[!, Symbol(xcol, "_mist")] .* HARTREE_TO_EV
    x_ref = df_stero[!, xcol] .* HARTREE_TO_EV
    y = df_stero[!, Symbol(ycol, "_mist")]
    y_ref = df_stero[!, ycol]
    r2_all_gap = cor(df[!, xcol], df[!, Symbol(xcol, "_mist")])
    r2_all_polar = cor(df[!, ycol], df[!, Symbol(ycol, "_mist")])


    @info "R2" cor(x, x_ref) cor(y, y_ref) r2_all_polar r2_all_gap

    markers = Dict(
        "OH" => :circle,
        "SP" => :+,
        "TB" => :utriangle,
    )

    f = Figure(; size=(160, 95))
    ax = Axis(f[1, 1];
        limits=((0, 6), (0, 500)),
        xlabel=L"$$HOMO-LUMO Gap (eV)",
        ylabel=L"Polarizability ($\AA$)",
    )

    for gdf in groupby(df_stero, :isomeric_smiles)
        xv = gdf[!, Symbol(xcol, "_mist")]
        xv = vcat(xv, xv[1])
        yv = gdf[!, Symbol(ycol, "_mist")]
        yv = vcat(yv, yv[1])
        lines!(ax, xv .* HARTREE_TO_EV, yv;
            color=:black,
            linestyle=:solid,
        )
    end

    h = scatter!(ax,
        df[!, Symbol(xcol, "_mist")] .* HARTREE_TO_EV,
        df[!, Symbol(ycol, "_mist")];
        color=levelcode.(df.metal),
        marker=map(x -> markers[x], df.stero),
        # alpha=0.5,
        markersize=2pt,
    )

    # Colorbar(f[1, 1], h;
    #     vertical=false,
    #     tellwidth=false,
    #     tellheight=false,
    #     valign=0.05,
    #     halign=0.05,
    #     width=50pt,
    #     size=5pt
    # )

    h = scatter!(ax, x, y;
        color=levelcode.(df_stero.metal),
        marker=map(x -> markers[x], df_stero.stero),
        strokewidth=0.5pt,
        colormap=:tab20,
        colorrange=(1, 20),
    )

    elems = map(enumerate(levels(df_stero.metal))) do (color, label)
        PolyElement(; label, color, colormap=h.colormap, colorrange=h.colorrange, strokewidth=0.5pt)
    end
    stero = map(levels(df_stero.stero)) do label
        MarkerElement(; label, marker=markers[label], color=:black, markersize=4pt)
    end

    Legend(f[:, end+1],
        [elems, stero],
        [MISTStyle.label.(elems), MISTStyle.label.(stero)],
        ["Metal", "Chirality"];
        orientation=:vertical,
        nbanks=2,
        tellwidth=true,
    )
    resize_to_layout!(f)


    return f
end

function figure_accuracy(df)
    df = transform(df, :smiles => ByRow(label_metal) => [:metal, :stero, :stero_id])
    subset!(df, :metal => ByRow(!isnothing), :stero => ByRow(!=("H")))
    @info unique(df.metal) unique(df.stero)
    @show combine(groupby(df, [:metal])) do gdf
        (;
            Polarizability=cor(gdf.Polarizability, gdf.Polarizability_mist),
            Gap=cor(gdf.HL_Gap, gdf.HL_Gap_mist),
            Dipole_M=cor(gdf.Dipole_M, gdf.Dipole_M_mist),
            n=nrow(gdf)
        )
    end
    return nothing
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
