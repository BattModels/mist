function hydrocarbon_trends(df; qm_model="")
    f = Figure(size=(2.9inch, 2inch))

    func_groups = [
        "Alkanes",
        "Alkenes",
        "Alkynes",
        # "Isoalkanes",
        "Arenes",
        # "Esters",
        # "Ethers",
        "Alcohols",
        # "Aldehydes",
        "Amines",
        "Nitriles",
        # "Dinitriles",
        "Carboxylic acids",
        # "Fluoroalkanes",
        # "Bromoalkanes",
        # "Chloroalkanes",
    ]
    df = subset(df, :type => ByRow(in(func_groups)))
    df.type = categorical(df.type; levels=func_groups)
    colormap = :tab10
    colorrange = (1, 10)

    # Trends with Size
    gl_trends = GridLayout(f[1, 1])
    axes = [
        Symbol("g298" * qm_model) => L"$G\degree$\n[eV]",
        Symbol("alpha" * qm_model) => L"$\alpha$\n$[\alpha_0^3]$",
        # Symbol("gap" * qm_model) => L"Gap\n[eV]$$",
        Symbol("homo" * qm_model) => L"HOMO\n[eV]$$",
        :mp => L"$$Melt\n[$\degree C$ ]",
        :bp => L"$$Boil\n[$\degree C$ ]",
        # Symbol("mu" * qm_model) => L"$\mu$ [D]",
        # Symbol("r2" * qm_model) => L"$\langle R^2 \rangle$\n$[\alpha_0^2]$",
        # :dn => L"$$DN\n[kJ/mol]",
        # :pKa_kt => L"pKa",
        # :alpha_kt => L"KT $\alpha$",
        # :beta_kt => L"KT $\beta$",
    ]
    axes = map(enumerate(axes)) do (idx, (col, ylabel))
        is_last = idx == length(axes)
        col => Axis(gl_trends[idx, 1];
            xlabel=L"$$Number of Carbons", ylabel,
            limits=((1, 25), nothing),
            xticks=LinearTicks(5),
            xlabelvisible=is_last,
            xticksvisible=is_last,
            xminorticksvisible=is_last,
            xticklabelsvisible=is_last,
            tellwidth=true,
            yticks=WilkinsonTicks(3),
            yminorticksvisible=true,
            xminorticks=IntervalsBetween(5),
        )
    end |> Dict
    foreach(groupby(df, :type)) do gdf
        for (col, ax) in pairs(axes)
            y = convert_units(gdf[:, col], col)
            lines!(ax, gdf.n_carbon, mean.(y);
                label=string(first(gdf.type)),
                color=levelcode.(gdf.type),
                colormap,
                colorrange,
            )
        end
    end
    elems = map(enumerate(func_groups)) do (color, label)
        PolyElement(; color, label, colormap, colorrange)
    end
    Legend(gl_trends[end+1, 1], elems, label.(elems);
        nbanks=4,
        tellheight=true, tellwidth=true,
    )

    # Design Rules
    gl_dr = GridLayout(f[1, 2])
    ax_dn = Axis(gl_dr[1, 1]; ylabel=L"$G\degree$ [eV]", xlabel=L"HOMO [eV]$$")
    ax_mp_bp = Axis(gl_dr[2, 1];
        xlabel=L"Melting Point [$\degree C$ ]",
        ylabel=L"Boiling Point [$\degree C$ ]",
    )
    foreach(groupby(df, :type)) do gdf
        kwargs = (;
            colormap,
            colorrange,
            color=levelcode.(gdf.type),
            label=string(first(gdf.type)),
            marker=:circle,
            # whiskerwidth=2pt,
            # markervisible=false,
        )
        homo = gdf[:, Symbol("homo" * qm_model)] .* HARTREE_TO_EV
        g298 = gdf[:, Symbol("g298" * qm_model)] .* HARTREE_TO_EV
        scatter!(ax_dn, mean.(homo), mean.(g298); kwargs...)
        scatter!(ax_mp_bp, mean.(gdf.mp), mean.(gdf.bp); kwargs...)
        # errorcross!(ax_dn, homo, g298; kwargs...)
        # errorcross!(ax_mp_bp, gdf.mp, gdf.bp; kwargs...)
    end
    h = ablines!(ax_mp_bp, 0, 1; color=:black, linestyle=:dash)
    MISTStyle.tantext!(ax_mp_bp, h, 25; text=L"T_m = T_b", align=(:center, :bottom))

    # Exceptions to BP > MP
    df_except = subset(df, [:mp, :bp] => ByRow((mp, bp) -> mean(mp) > mean(bp)))
    @info "Exceptions to BP > MP" df_except[:, [:type, :smi, :mp, :bp]]

    colgap!(f.layout, 5)
    colsize!(f.layout, 1, Relative(2 / 3))
    sublabel!(f[1, 1, TopLeft()], "a"; left=25)
    sublabel!(gl_dr[1, 1, TopLeft()], "b"; left=3)
    sublabel!(gl_dr[2, 1, TopLeft()], "c"; left=3)
    resize_to_layout!(f)

    return f
end


function electrolyte_trends(df)
    f = Figure(size=(3.5inch, 1.7inch))
    df = deepcopy(df)

    func_groups = [
        "Ethylene carbonate", "Carbonate ester", "Ester", "Ether",
        "Dicarbonate", "Alkane", "Alkene", "Arene",
    ]
    subset!(df, :class => ByRow(in(func_groups)))
    df.class = categorical(df.class; levels=func_groups)


    func_groups_mods = Dict(
        "Baseline" => :+,
        "Chloro" => :pentagon,
        "Fluoro" => :hexagon,
        "Sulfone" => :diamond,
        "Phosphate" => :star5,
        "Silyl" => :rect,
        "Nitrile" => :utriangle,
    )
    df.subsitute .= coalesce.(df.subsitute, "Baseline")
    subset!(df, :subsitute => ByRow(in(collect(keys(func_groups_mods)))))
    marker = map(df.subsitute) do subsitute
        return func_groups_mods[subsitute]
    end
    color = map(class -> MISTStyle.CAT_COLORS[levelcode(class)], df.class)

    @info sort(combine(nrow, groupby(df, [:class, :subsitute])), :class)

    marker_elems = map(collect(pairs(func_groups_mods))) do (label, marker)
        label = coalesce(label, "Baseline")
        MarkerElement(; label, marker)
    end
    color_elems = map(enumerate(func_groups)) do (idx, label)
        PolyElement(; color=MISTStyle.CAT_COLORS[idx], label)
    end
    gl = GridLayout(f[1, 1]; default_colgap=3)
    Legend(f[1, 2],
        [marker_elems, color_elems],
        [label.(marker_elems), label.(color_elems)],
        ["Substitutions", "Functional Groups"];
        tellwidth=true,
        tellheight=true,
        valign=:top,
        margin=(0, 0, 0, 0),
    )

    ax = Axis(gl[1, 1]; ylabel=L"DN [kcal/mol, BF3]$$", xlabel=L"HOMO [eV]$$")
    scatter!(ax, mean.(df.homo) .* HARTREE_TO_EV, mean.(df.dn .* JOULES_TO_CALORIES); marker, color)

    ax = Axis(gl[1, 2];
        ylabel=L"DN [kcal/mol, BF3]$$", xlabel=L"KT $\beta$",
        limits=((0, nothing), nothing),
    )
    beta_kt = 2 .* sigmoid.(mean.(df.beta_kt))
    scatter!(ax, beta_kt, mean.(df.dn .* JOULES_TO_CALORIES); marker, color)


    ax = Axis(gl[1, 3]; ylabel=L"$\mu$ [D]", xlabel=L"Minimum Partial Charge$$")
    scatter!(ax, df.min_lowdin, mean.(df.pKa_kt); marker, color)

    ax = Axis(gl[2, 1]; xlabel=L"Boiling Point $[\degree C ]$", ylabel=L"Flash Point $[\degree C ]$")
    # Fit coefficients from 10.1149/2.0121502jes
    b = 0.7099
    a = 26.62
    C_TO_K = 273.15
    a = a + (b * C_TO_K) - C_TO_K
    scatter!(ax, mean.(df.bp), mean.(df.fp); marker, color)
    h = ablines!(ax, a, b; color=MISTStyle.UM_COLORS.blue, label="Hess et al.")
    axislegend(ax, position=:rb, padding=(1, 1, 1, 1), margin=(1, 1, 1, 1))

    ax = Axis(gl[2, 2]; xlabel=L"Melting Point $[\degree C]$", ylabel=L"Boiling Point $[\degree C ]$")
    quadrant!(ax, 0, 75, :lt)

    scatter!(ax, mean.(df.mp), mean.(df.bp); marker, color)
    tantext!(ax, ablines!(ax, 0, 1; color=:black, linestyle=:dash), -2;
        text=L"T_m = T_b",
        align=(:left, :top),
    )

    ax = Axis(gl[2, 3];
        xlabel=L"HOMO [eV]$$", ylabel=L"Gap [eV]$$",
        limits=((-11, -5.5), (3, 10.75)),
    )
    @show dme_homo = mean(first(df[df[!, :name].=="DME", :homo]) * HARTREE_TO_EV)
    quadrant!(ax, dme_homo, 5.0, :lt)
    scatter!(ax, mean.(df.homo_rand) .* HARTREE_TO_EV, mean.(df.gap_rand) .* HARTREE_TO_EV; marker, color)

    resize_to_layout!(f)

    return f
end

function convert_units(y, col)
    if endswith(string(col), "_rand")
        col = Symbol(string(col)[1:end-5])
    end
    if col in [:homo, :lumo, :gap, :zpve, :u0, :g298, :u298, :h298]
        y .*= HARTREE_TO_EV
    elseif col in [:dn]
        y .*= JOULES_TO_CALORIES
    end
    return y
end

function figure_permutations(name_df::Pair...; name_df_order)
    f = Figure(size=(3.42inch, 2inch), figure_padding=(2, 3, 2, 4))
    gl_order = GridLayout(f[1, 1])
    gl_trends = GridLayout(f[1, 2])

    axes = [
        :homo => L"HOMO$$",
        :gap => L"Gap$$",
        :lumo => L"LUMO$$",
        :zpve => L"ZPVE$$",
        :g298 => L"$G\degree$",
    ]
    limits = Dict(
        :homo => (nothing, (3e-3, 2)),
        :gap => (nothing, (5e-3, 2)),
        :zpve => (nothing, (5e-3, 3e-1)),
    )
    dfs = []
    for (name, df) in name_df
        df = deepcopy(df)
        df._plt_name .= name
        push!(dfs, df)
    end
    df = vcat(dfs...)
    df.type = categorical(df.type)
    df._plt_name = categorical(df._plt_name)

    axes = map(enumerate(axes)) do (idx, (col, ylabel))
        is_last = idx == length(axes)
        ax_tend = Axis(gl_trends[idx, 1];
            ylabel,
            yscale=log10,
            xticklabelrotation=0.55,
            xticks=MISTStyle.categorical_ticks(df.type),
            xlabelvisible=is_last,
            xticksvisible=is_last,
            xminorticksvisible=is_last,
            xticklabelsvisible=is_last,
            tellwidth=true,
            yticks=LogTicks(WilkinsonTicks(3)),
            yminorticksvisible=true,
            xminorticks=IntervalsBetween(5),
        )
        ax_scatter = Axis(gl_trends[idx, 2];
            xscale=ax_tend.yscale,
            yscale=ax_tend.yscale,
            xlabel="Comparative\nRobustness",
            xlabelvisible=is_last,
            xticksvisible=false,
            xticklabelsvisible=false,
            yticksvisible=false,
            yticklabelsvisible=false,
        )
        return col => (ax_tend, ax_scatter)
    end |> Dict

    for (col, (ax, axs)) in axes
        y = std.(df[:, col])
        y = convert_units(y, col)
        x = df.type
        dodge = levelcode.(df._plt_name)
        idx = @. !isnan(y)
        x = x[idx]
        y = y[idx]
        dodge = dodge[idx]
        boxplot!(ax, levelcode.(x), y;
            dodge,
            show_outliers=false,
            color=dodge,
            colormap=MISTStyle.CAT_COLORS,
            colorrange=(1, length(MISTStyle.CAT_COLORS)),
        )
        scatter!(axs, y[dodge.==2], y[dodge.==1];
            color=MISTStyle.UM_COLORS.blue,
            alpha=0.2,
            marker=:circle,
        )
        powerlaw!(axs, 1, 1; color=:black)
        if haskey(limits, col)
            ax.limits[] = limits[col]
            axs.limits[] = (last(limits[col]), last(limits[col]))
        end
        linkyaxes!(axs, ax)
    end

    # Legend
    elements = map(enumerate(levels(df._plt_name))) do (i, label)
        PolyElement(
            color=MISTStyle.CAT_COLORS[i],
            label=label
        )
    end
    Legend(gl_trends[end, 1], elements, MISTStyle.label.(elements);
        orientation=:horizontal,
        halign=:left,
        valign=:top,
        alignmode=Outside(),
    )

    colsize!(gl_trends, 1, Relative(3 / 4))
    colgap!(gl_trends, 1, 2)

    figure_double_bond_loc!(gl_order, name_df...; name_df_order)


    sublabel!(gl_order[2, 1, TopLeft()], "a"; left=15)
    sublabel!(gl_order[3, 1, TopLeft()], "b"; left=15)
    sublabel!(gl_trends[1, 1, TopLeft()], "c"; left=13)

    resize_to_layout!(f)


    return f
end

function figure_double_bond_loc(name_df::Pair...; name_df_order)
    f = Figure(; size=(113, 136), figure_padding=(2,4,2,2))
    return figure_double_bond_loc!(f, name_df...; name_df_order)
end
function figure_double_bond_loc!(f, name_df::Pair...; name_df_order)
    # Order Sensitivity
    n_carbon_range = extrema(last(first(name_df_order)).n_carbon)
    cb = Colorbar(f[1, 1];
        label="Number of Carbons",
        colorrange=n_carbon_range,
        vertical=false, tellwidth=false,
    )
    axes = Axis[]
    for (idx, (name, df)) in enumerate(name_df_order)
        is_last = idx == length(name_df_order)
        ax = Axis(f[1+idx, 1];
            xlabel=L"Double Bond Location$$",
            limits=((0, 1), nothing),
            xtickformat="{:.0%}",
            xlabelvisible=is_last,
            xticksvisible=is_last,
            xticklabelsvisible=is_last,
            yticks=WilkinsonTicks(5),
        )
        text!(0.99, 0.0;
            text=name,
            align=(:right, :bottom),
            space=:relative,
        )
        push!(axes, ax)
        df = subset(df, :n_carbon => ByRow(>(4)))
        foreach(groupby(df, :n_carbon)) do gdf
            n_carbon = gdf.n_carbon[1]
            homo = gdf.homo .* HARTREE_TO_EV
            h = lines!(ax, gdf.rel_pos, mean.(homo);
                color=n_carbon,
                MISTStyle.cb_attrs(cb)...
            )
            lb = mean.(homo) .- stderror.(homo)
            ub = mean.(homo) .+ stderror.(homo)
            band!(ax, gdf.rel_pos, lb, ub;
                alpha=0.2,
                color=h.color,
                MISTStyle.cb_attrs(cb)...
            )
        end
    end
    Label(f[:, 0]; text=L"HOMO [eV]$$", rotation=pi/2)


    linkyaxes!(axes...)
    resize_to_layout!(f)


    return f
end

function figure_order2(name_df::Pair...)
    f = Figure(; size=(3.42inch, 2inch), figure_padding=(2, 3, 2, 2))
    n_carbon_range = extrema(last(first(name_df)).n_carbon)
    cb = Colorbar(f[2, 1:length(name_df)];
        label="Number of Carbons",
        colorrange=n_carbon_range,
        vertical=false,
        flipaxis=false,
        height=5
    )
    for (idx, (name, df)) in enumerate(name_df)
        is_first = idx == 1
        ax = Axis(f[1, idx];
            xlabel=L"Location of Double Bond$$",
            ylabel=L"HOMO [eV]$$",
            limits=((0, 1), nothing),
            ylabelvisible=is_first,
            yticksvisible=is_first,
            yticklabelsvisible=is_first,
        )
        df = subset(df, :n_carbon => ByRow(>(4)))
        df = subset(df, :n_carbon => ByRow(n -> n % 2 == 0))
        foreach(groupby(df, :n_carbon)) do gdf
            n_carbon = gdf.n_carbon[1]
            homo = gdf.homo .* HARTREE_TO_EV
            h = lines!(ax, gdf.rel_pos, mean.(homo);
                color=n_carbon,
                MISTStyle.cb_attrs(cb)...
            )
            lb = mean.(homo) .- stderror.(homo)
            ub = mean.(homo) .+ stderror.(homo)
            band!(ax, gdf.rel_pos, lb, ub;
                alpha=0.2,
                color=h.color,
                MISTStyle.cb_attrs(cb)...
            )
        end
    end
    return f
end

function figure_fatty_acids(df; omega=3, alpha=0.8, qm_model="")
    f = Figure(size=(3.42inch, 2inch))

    gl_trends = GridLayout(f[1, 1])
    gl_cross = GridLayout(f[1, 2])
    sublabel!(f[1, 1, TopLeft()], "a"; left=5)
    sublabel!(f[1, 2, TopLeft()], "b"; left=5)

    d_max = maximum(df.d)
    n_max = maximum(df.n)
    df_sat = subset(df, :d => ByRow(==(0)))
    df_unsat = subset(df, :d => ByRow(!=(0)), :n => ByRow(==(omega)))

    # Trends with ω-n
    axes = [
        Symbol("g298" * qm_model) => L"$G\degree$\n[eV]",
        Symbol("alpha" * qm_model) => L"$\alpha$\n$[\alpha_0^3]$",
        # Symbol("alpha" * qm_model) => L"$\alpha$\n$[\alpha_0^3]$",
        # Symbol("gap" * qm_model) => L"Gap\n[eV]$$",
        :mp => L"$$Melt\n[$\degree C$ ]",
        :bp => L"$$Boil\n[$\degree C$ ]",
        :fp => L"$$Flash\n[$\degree C$ ]",
    ]
    axes = map(enumerate(axes)) do (idx, (col, ylabel))
        is_last = idx == length(axes)
        col => Axis(gl_trends[idx, 1];
            xlabel=L"Chain Length$$", ylabel,
            xlabelvisible=is_last,
            xticksvisible=is_last,
            xticklabelsvisible=is_last,
            xminorticksvisible=is_last,
            xticks=0:5:25,
            xminorticks=IntervalsBetween(5),
            tellwidth=true,
            yticks=WilkinsonTicks(3),
            yminorticksvisible=true,
        )
    end |> Dict
    cb = Colorbar(gl_trends[0, 1];
        colorrange=(0, d_max),
        label="Number of Double Bonds",
        vertical=false, tellwidth=false,
    )
    local h, hs
    for (col, ax) in pairs(axes)
        foreach(groupby(df_unsat, :n)) do gdf
            y = convert_units(gdf[:, col], col)
            hs = scatter!(ax, gdf.c, mean.(y);
                color=gdf.d,
                marker=:circle,
                alpha=0.8,
                MISTStyle.cb_attrs(cb, BoxPlot)...
            )
        end
        h = lines!(ax, df_sat.c, mean.(convert_units(df_sat[:, col], col));
            color=:black,
            linewidth=1,
            label="Saturated",
            MISTStyle.cb_attrs(cb, BoxPlot)...
        )
    end
    elems = [h, MarkerElement(color=:black, label="Unsaturated", marker=hs.marker, markersize=hs.markersize)]
    Legend(gl_trends[end, 1], elems, label.(elems);
        alignmode=Inside(),
        labelsize=6pt,
        halign=:right,
        valign=:bottom,
        nbanks=2,
    )

    n = 2
    ax_fp_c = Axis(gl_cross[1, 1];
        xlabel=L"Number of Carbons$$",
        ylabel=L"Flash Point [$\degree C$ ]",
        xlabelvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(5),
    )
    ax_fp_mu = Axis(gl_cross[1, 2];
        xlabel=L"$\alpha$ [$\alpha_0^3$]",
        ylabel=L"Flash Point [$\degree C$ ]",
        xlabelvisible=false,
        xticksvisible=false,
        xticklabelsvisible=false,
        ylabelvisible=false,
        yticksvisible=false,
        yticklabelsvisible=false,
    )
    ax_mp_c = Axis(gl_cross[2, 1];
        xlabel=L"Chain Length$$",
        ylabel=L"Melting Point [$\degree C$ ]",
        xticks=[0, 10, 20],
        xminorticks=IntervalsBetween(5),
        xminorticksvisible=true,
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(5),
    )
    ax_mp_mu = Axis(gl_cross[2, 2];
        xlabel=L"$\alpha$ [$\alpha_0^3$]",
        ylabel=L"Melting Point [$\degree C$ ]",
        ylabelvisible=false,
        yticksvisible=false,
        yticklabelsvisible=false,
        xticks=WilkinsonTicks(3),
        xminorticksvisible=true,
    )
    linkyaxes!(ax_fp_c, ax_fp_mu)
    linkyaxes!(ax_mp_c, ax_mp_mu)
    linkxaxes!(ax_fp_c, ax_mp_c)
    linkxaxes!(ax_fp_mu, ax_mp_mu)
    cb_sat = Colorbar(gl_cross[1:n, 3];
        label="Degree of Saturation",
        tickformat="{:.0%}",
        colorrange=extrema(df.saturation),
    )

    sort!(df_unsat, :saturation)
    alpha_chn = mean.(df_unsat[:, Symbol("mu" * qm_model)])
    scatter!(ax_fp_c, df_unsat.c, mean.(df_unsat.fp);
        color=df_unsat.saturation,
        marker=:circle,
        alpha,
        MISTStyle.cb_attrs(cb_sat, Scatter)...
    )
    scatter!(ax_fp_mu, alpha_chn, mean.(df_unsat.fp);
        color=df_unsat.saturation,
        marker=:circle,
        alpha,
        MISTStyle.cb_attrs(cb_sat, Scatter)...
    )

    scatter!(ax_mp_c, df_unsat.c, mean.(df_unsat.mp);
        color=df_unsat.saturation,
        marker=:circle,
        alpha,
        MISTStyle.cb_attrs(cb_sat, Scatter)...
    )
    scatter!(ax_mp_mu, alpha_chn, mean.(df_unsat.mp);
        color=df_unsat.saturation,
        marker=:circle,
        alpha,
        MISTStyle.cb_attrs(cb_sat, Scatter)...
    )

    colsize!(f.layout, 1, Relative(0.6))
    resize_to_layout!(f)

    return f
end
