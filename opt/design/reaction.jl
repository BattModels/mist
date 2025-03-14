
# include("MIST.jl")
# using .MIST

using PythonCall
using DataFrames
using Makie
using CategoricalArrays
using ManifoldLearning


steps = [
    "CC(C)(CO)C(C(=O)NCCC(=O)O)O",
    "CC(C)(COP(=O)(O)O)[C@H](C(=O)NCCC(=O)O)O",
    "CC(C)(COP(=O)(O)O)[C@H](C(=O)NCCC(=O)N[C@@H](CS)C(=O)O)O",
    "O=C(NCCS)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)O",
    "CC(C)(COP(=O)(O)OP(=O)(O)OC[C@@H]1[C@H]([C@H]([C@@H](O1)N2C=NC3=C(N=CN=C32)N)O)O)[C@H](C(=O)NCCC(=O)NCCS)O",
    "O=C(NCCS)CCNC(=O)C(O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n2cnc1c(ncnc12)N)[C@H](O)[C@@H]3OP(=O)(O)O",
]

function predict_pathway(steps, model, tok)
    o = model.predict(PyList(steps), tok)
    G = pyconvert(Vector{Float64}, o["g298"])
    f = Figure()
    ax = Axis(f[1, 1]; ylabel="Gibbs Free Energy @ 298K [Ha]")
    stairs!(ax, G, step=:center)
    f
end

function alkene_chain_gen(n::Int)
    smi = []
    for len in range(3, n)
        for pos in range(1, len - 1; step=1)
            push!(smi, (;
                len,
                pos,
                smi=("C"^pos) * "=" * "C"^(len - pos),
            ))
        end
    end
    return DataFrame(smi)
end

function plot_alkene_chain(n, model, tok)
    df = alkene_chain_gen(n)
    df.type .= "normal"
    df2 = copy(df)
    df2.smi = map(smi -> replace(smi, "=" => "/C=C/"), df2.smi)
    df2.len .+= 2
    df2.type .= "cis"
    df3 = copy(df)
    df3.smi = map(smi -> replace(smi, "=" => "/C=C\\"), df2.smi)
    df3.len .+= 2
    df3.type .= "trans"
    df = vcat(df, df2, df3)

    pred = model.predict(PyList(df.smi), tok)
    df.h298 = pyconvert(Vector{Float64}, pred["h298"])
    df.mu = pyconvert(Vector{Float64}, pred["mu"])

    f = Figure()
    ax = Axis(f[1, 1];
        # limits=((0, 1), nothing),
        # yscale=log10,
        ylabel="H298 [kcal/mol]")
    cb = Colorbar(f[1, 2], limits=(3, n + 2), colormap=:viridis, label="Length")
    for len in unique(df.len)
        for (type, marker) in [("normal", :x), ("cis", :cross), ("trans", :diamond)]
            df_len = subset(df, :len => ByRow(==(len)), :type => ByRow(==(type)))
            scatter!(ax,
                range(1, nrow(df_len)),
                df_len.h298;
                color=len,
                marker,
                colorrange=cb.limits,
                colormap=cb.colormap,
            )
        end
    end
    return f, df
end

alkyne(n::Int) = "C#" * "C"^(n-1)
alkene(n::Int) = "C=" * "C"^(n-1)
alkane(n::Int) = "C" ^ n
isoalkane(n::Int) = "C(C)" * "C" ^ (n - 2)
alcohol(n::Int) = "O" * "C" ^ n
aldehyde(n::Int) = "O=" * "C"^n
nitrile(n::Int) = "N#" * "C"^n
dinitrile(n::Int) = "N#" * "C"^n * "#N"
amine(n::Int) = "N" * "C"^ n
carboxylic_acid(n::Int) = "C(=O)" * "C"^(n-1)
halide(n::Int, element::String) = element * "C"^n
halide(element::String) = Base.Fix2(halide, element)
tetra_sub_alkene(n::Int) = "$("C"^n)C(=C($("C"^n))$("C"^n))$("C"^n)"
disubstituted_alkyne(n::Int) = "$("C"^n)#$("C"^n)"
triboroester(n::Int) = "O(B(O$("C"^n))O$("C"^n))$("C"^n)"
trialkylborane(n::Int) = "B($("C"^n))($("C"^n))$("C"^n)"

function carbon_chains(n::Int)
    return DataFrame(vcat(
        [(; n, type="Alkanes", smi=alkane(n)) for n in 1:n],
        # [(; n, type="Isoalkanes", smi=isoalkane(n)) for n in 3:n],
        [(; n, type="Alcohols", smi=alcohol(n)) for n in 1:n],
        [(; n, type="Nitrile", smi=nitrile(n)) for n in 1:n],
        # [(; n, type="Dinitriles", smi=dinitrile(n)) for n in 1:n],
        [(; n, type="Amines", smi=amine(n)) for n in 1:n],
        [(; n, type="Carboxylic Acids", smi=carboxylic_acid(n)) for n in 2:n],
        [(; n, type="Fluoroalkanes", smi=halide(n, "F")) for n in 2:n],
        # [(; n, type="Bromoalkanes", smi=halide(n, "Br")) for n in 2:n],
        # [(; n, type="Chloroalkanes", smi=halide(n, "Cl")) for n in 2:n],
        [(; n, type="Tetra subsitued Alkenes", smi=tetra_sub_alkene(n)) for n in 1:cld(n, 4)],
        [(; n, type="Disubsitued Alkynes", smi=disubstituted_alkyne(n)) for n in 1:cld(n, 2)],
    ))
end

function melt_boil_trends_monte(n::Int, model)
    alkanes = [alkane(n) for n in 1:n]
    isoalkanes = [isoalkane(n) for n in 3:n]

    df_alkanes = MIST.predict_monte(alkanes, model; n=50)
    df_alkanes.n = 1:n
    df_isoalkanes = MIST.predict_monte(isoalkanes, model; n=50)
    df_isoalkanes.n = 3:n

    f = Figure()
    ax = Axis(f[1, 1]; ylabel="Boiling Point [°C]")
    errorbars!(ax, df_alkanes.n, df_alkanes.bp_mean, df_alkanes.bp_stderr; label="Alkanes")
    errorbars!(ax, df_isoalkanes.n, df_isoalkanes.bp_mean, df_isoalkanes.bp_stderr; label="Isoalkanes")
    axislegend(ax; position=:rb)

    ax = Axis(f[2, 1]; ylabel="Melting Point [°C]")
    errorbars!(ax, df_alkanes.n, df_alkanes.mp_mean, df_alkanes.mp_stderr; label="Alkanes")
    errorbars!(ax, df_isoalkanes.n, df_isoalkanes.mp_mean, df_isoalkanes.mp_stderr; label="Isoalkanes")

    return f
end

function predict_class(class::Dict, model)
    smi = Iterators.flatten(values(class))
    df = MIST.predict(collect(smi), model)
    types = map((kv) -> repeat([kv[1]], length(kv[2])), collect(pairs(class)))
    df.types = collect(Iterators.flatten(types))
    return df
end



function melt_boil_trends(n::Int, model)
    alkanes =
    isoalkanes = [(; n, type="isoalkane", smi=isoalkane(n)) for n in 3:n]
    alchols = [(; n, type="isoalkane", smi=isoalkane(n)) for n in 3:n]
    df = DataFrame(vcat(
        [(; n, type="Alkanes", smi=alkane(n)) for n in 1:n],
        [(; n, type="Isoalkane", smi=isoalkane(n)) for n in 3:n],
        [(; n, type="Alchols", smi=alchol(n)) for n in 1:n],
    ))
    return df




    model = model.eval()
    df_alkanes = MIST.predict(alkanes, model)
    df_alkanes.n = 1:n
    df_isoalkanes = MIST.predict(isoalkanes, model)
    df_isoalkanes.n = 3:n

    f = Figure()
    ax_bp = Axis(f[1, 1]; ylabel="Boiling Point [°C]")
    ax_mp = Axis(f[2, 1]; ylabel="Melting Point [°C]")
    for type in unique(df.type)
        df_type = subset(df, :type => ByRow(==(type)))
        lines!(ax_bp, df.n, df.bp; label=type)
        lines!(ax_mp, df.n, df.bp; label=type)
    end
    axislegend(ax_bp; position=:rb)

    return f
end

function dipole_trends(n::Int; qm, dn, kt, mpbp, samples=20)
    df = carbon_chains(n)
    df_qm9 = MIST.predict_monte(df.smi, qm; n=samples)
    select!(df_qm9, Not(:smi))
    df_dn = MIST.predict_monte(df.smi, dn; n=samples)
    select!(df_dn, "BF3 affinity_mean" => :donor_number_mean, "BF3 affinity_stderr" => :donor_number_stderr)
    df_mpbp = MIST.predict_monte(df.smi, mpbp; n=samples)
    select!(df_mpbp, :mp_mean, :mp_stderr, :bp_mean, :bp_stderr)
    df_kt = MIST.predict_monte(df.smi, kt; n=samples)
    select!(df_kt, Not(:smi))
    rename!(df_kt,
        "alpha_mean" => :alpha_kt_mean,
        "alpha_stderr" => :alpha_kt_stderr,
        "beta_mean" => :beta_kt_mean,
        "beta_stderr" => :beta_kt_stderr,
    )
    df = hcat(df, df_qm9, df_dn, df_kt, df_mpbp)

    f = Figure()
    gl_trends = GridLayout(f[1,1])
    rowgap!(gl_trends, 5)
    ax = Axis(gl_trends[1, 1]; ylabel=L"$\mu$ [D]")
    hidexdecorations!(ax)
    for type in unique(df.type)
        df_type = subset(df, :type => ByRow(==(type)))
        sort!(df_type, :n)
        h = lines!(ax, df_type.n, df_type.mu_mean; label=type)
        errorbars!(ax, df_type.n, df_type.mu_mean, df_type.mu_stderr; color=h.color)
    end

    ax = Axis(gl_trends[2, 1]; ylabel=L"$G^{\degree}$ [kJ/mol]")
    hidexdecorations!(ax)
    for type in unique(df.type)
        df_type = subset(df, :type => ByRow(==(type)))
        sort!(df_type, :n)
        h = lines!(ax, df_type.n, df_type.h298_mean; label=type)
        errorbars!(ax, df_type.n, df_type.h298_mean, df_type.h298_stderr; color=h.color)
    end

    ax = Axis(gl_trends[3, 1]; ylabel=L"DN", xlabel=L"$$Number of Carbons")
    hidexdecorations!(ax)
    for type in unique(df.type)
        df_type = subset(df, :type => ByRow(==(type)))
        sort!(df_type, :n)
        h = lines!(ax, df_type.n, df_type.donor_number_mean; label=type)
        errorbars!(ax, df_type.n, df_type.donor_number_mean, df_type.donor_number_stderr; color=h.color)
    end

    ax = Axis(gl_trends[4, 1]; ylabel=L"$\alpha$ [$\alpha_0^2$]", xlabel=L"$$Number of Carbons")
    hidexdecorations!(ax)
    for type in unique(df.type)
        df_type = subset(df, :type => ByRow(==(type)))
        sort!(df_type, :n)
        h = lines!(ax, df_type.n, df_type.alpha_mean; label=type)
        errorbars!(ax, df_type.n, df_type.alpha_mean, df_type.alpha_stderr; color=h.color)
    end

    ax = Axis(gl_trends[5, 1]; ylabel=L"KT $\beta$", xlabel=L"$$Number of Carbons")
    hidexdecorations!(ax)
    for type in unique(df.type)
        df_type = subset(df, :type => ByRow(==(type)))
        sort!(df_type, :n)
        h = lines!(ax, df_type.n, df_type.beta_kt_mean; label=type)
        errorbars!(ax, df_type.n, df_type.beta_kt_mean, df_type.beta_kt_stderr; color=h.color)
    end

    ax = Axis(gl_trends[6, 1]; ylabel=L"Melting Point [$^{°}C$]", xlabel=L"$$Number of Carbons")
    for type in unique(df.type)
        df_type = subset(df, :type => ByRow(==(type)))
        sort!(df_type, :n)
        h = lines!(ax, df_type.n, df_type.mp_mean; label=type)
        errorbars!(ax, df_type.n, df_type.mp_mean, df_type.mp_stderr; color=h.color)
    end

    gl_corr = GridLayout(f[1,2])
    ax = Axis(gl_corr[1, 1]; ylabel=L"DN", xlabel=L"Kamlet-Taft $\beta$")
    for type in unique(df.type)
        df_type = subset(df, :type => ByRow(==(type)))
        h = errorbars!(ax, df_type.beta_kt_mean, df_type.donor_number_mean, df_type.donor_number_stderr, label=type)
        errorbars!(ax, df_type.beta_kt_mean, df_type.donor_number_mean, df_type.beta_kt_stderr; direction=:x, color=h.color)
    end

    ax = Axis(gl_corr[2, 1]; ylabel=L"$\alpha$ [$a_0^3$]", xlabel=L"$\mu$ [D]")
    for type in unique(df.type)
        df_type = subset(df, :type => ByRow(==(type)))
        h = errorbars!(ax, df_type.mu_mean, df_type.alpha_mean, df_type.alpha_stderr, label=type)
        errorbars!(ax, df_type.mu_mean, df_type.alpha_mean, df_type.mu_stderr; direction=:x, color=h.color)
    end

    Legend(f[2, :], ax; nbanks=2, tellheight=true)
    rowgap!(gl_trends, 5)
    resize_to_layout!(f)

    return f
end

labels(x) = x.label[]

function hard_soft_acid(; qm, kt, samples=20)
    df = DataFrame(vcat(
        [(; smi, type="Hard Acid") for smi in [
            "FB(F)F", "O=S(=O)=O", "OB(O)O", "O(B(OC)OC)C",
            map(triboroester, 1:5)...,
        ]],
        [(; smi, type="Borderline Acid") for smi in [
            map(aldehyde, 1:2:10)...,
        ]],
        [(; smi, type="Soft Acid") for smi in [
            map(aldehyde, 1:5:30)...,
        ]],
        [(; smi, type="Hard Base") for smi in [
            map(amine, 1:5:30)...,
            map(alcohol, 1:5:30)...,
        ]],
        [(; smi, type="Soft Base") for smi in [
            map(alkene, 1:5:30)...,
            map(tetra_sub_alkene, 1:10)...,
            map(disubstituted_alkyne, 1:3:15)...,
        ]],
    ))
    df_qm9 = MIST.predict_monte(df.smi, qm; n=samples)
    select!(df_qm9, Not(:smi))
    df_kt = MIST.predict_monte(df.smi, kt; n=samples)
    select!(df_kt, :pKa_mean, :pKa_stderr)
    df = hcat(df, df_qm9, df_kt)
    df.type = categorical(df.type)
    color = map(type -> MISTStyle.CAT_COLORS[levelcode(type)], df.type)


    f = Figure()
    ax = Axis(f[1, 1]; xlabel=L"$\alpha$ [$\alpha^2_0$]", ylabel=L"$$pKa")
    xyerrorbars!(ax, df.alpha_mean, df.pKa_mean, df.alpha_stderr, df.pKa_stderr; label=df.type, color)

    ax = Axis(f[2, 1]; xlabel=L"$$LUMO [Har]", ylabel=L"$$pKa")
    h = xyerrorbars!(ax, df.lumo_mean, df.pKa_mean, df.lumo_stderr, df.pKa_stderr; label=df.type, color)

    ax = Axis(f[1, 2])
    hidexdecorations!(ax)
    hideydecorations!(ax)
    emb = MIST.embed(df.smi, qm)
    m = fit(TSNE, emb'; p=30, maxoutdim=2)
    r = predict(m)
    scatter!(ax, r[1, :], r[2, :]; color)

    types = map(enumerate(levels(df.type))) do (i, label)
        PolyElement(
            color=MISTStyle.CAT_COLORS[i],
            label=label
        )
    end

    Legend(f[2, 2], types, labels.(types))

    return f


end

function xyerrorbars!(ax, x, y, errorx, errory; label=nothing, kwargs...)
    h = errorbars!(ax, x, y, errory; label, kwargs...)
    errorbars!(ax, x, y, errorx; direction=:x, color=h.color)
    return h
end
