function hydrocarbon_trends(df)
    f = Figure(size=(3.5inch, 3inch))
    # Trends with Size
    gl_trends = GridLayout(f[1, 1])
    axes = [
        :u298 => L"$G\degree$\n[kJ/mol]",
        :mu => L"$\mu$ [D]",
        :r2 => L"$\langle R^2 \rangle$\n$[\alpha_0^2]$",
        :gap => L"Gap\n[eV]$$",
        :mp => L"$$Melt\n[$\degree C$]",
        :bp => L"$$Boil\n[$\degree C$]",
        # :dn => L"$$DN\n[kJ/mol]",
        :pKa_kt => L"pKa",
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
            yticks=WilkinsonTicks(5),
            yminorticksvisible=true,
            xminorticks=IntervalsBetween(5),
        )
    end |> Dict
    foreach(groupby(df, :type)) do gdf
        for (col, ax) in pairs(axes)
            y = gdf[:, col]
            if col in [:gap]
                y .*= HARTREE_TO_EV
            end
            errorlines!(ax, gdf.n_carbon, y; label=string(first(gdf.type)))
        end
    end
    Legend(gl_trends[length(axes)+1, 1], axes[:mu]; nbanks=3, tellheight=true)

    # Design Rules
    gl_dr = GridLayout(f[1, 2])
    ax_dn = Axis(gl_dr[1, 1]; ylabel=L"DN", xlabel=L"HOMO [eV]$$")
    ax_alpha_beta = Axis(gl_dr[2, 1]; ylabel=L"KT $\beta$", xlabel=L"KT $\alpha$")
    foreach(groupby(df, :type)) do gdf
        label = string(first(gdf.type))
        errorcross!(ax_dn, HARTREE_TO_EV .* gdf.homo, gdf.dn; label)
        errorcross!(ax_alpha_beta, gdf.alpha_kt, gdf.beta_kt; label)
    end

    colgap!(f.layout, 5)
    colsize!(f.layout, 1, Relative(2 / 3))
    resize_to_layout!(f)

    return f
end

function electrolyte_trends(df)
    f = Figure(size=(3.5inch, 3inch))

    ax = Axis(f[1, 1]; ylabel=L"DN$$", xlabel=L"HOMO [eV]$$")
    errorcross!(ax, df.homo .* HARTREE_TO_EV, df.dn)
    vlines!(ax, -11.444; color=:black) # 10.1021/jz500485r
    hlines!(ax, 10; color=:black) # 10.1021/acsenergylett.3c00004

    ax = Axis(f[2, 1]; ylabel=L"DN$$", xlabel=L"KT $\beta$")
    errorcross!(ax, df.beta_kt, df.dn)
    hlines!(ax, 10; color=:black) # 10.1021/acsenergylett.3c00004

    ax = Axis(f[3, 1]; xlabel=L"HOMO [eV]$$", ylabel=L"Gap [eV]$$")
    errorcross!(ax, df.homo .* HARTREE_TO_EV, df.gap .* HARTREE_TO_EV)
    vlines!(ax, -11.444; color=:black) # 10.1021/jz500485r
    hlines!(ax, 5; color=:black) # 10.1021/acsenergylett.3c00004 (Really just says 5eV is good

    ax = Axis(f[1, 2]; xlabel=L"Melting Point $[\degree C]$", ylabel=L"Flash Point $[\degree C]$")
    vlines!(ax, -100; color=:black)
    hlines!(ax, 60; color=:black)
    errorcross!(ax, df.mp, df.fp)

    ax = Axis(f[2, 2]; xlabel=L"Melting Point $[\degree C]$", ylabel=L"Boiling Point $[\degree C]$")
    vlines!(ax, -100; color=:black)
    hlines!(ax, 60; color=:black)
    errorcross!(ax, df.mp, df.bp)


    return f
end

function convert_units(y, col)
    if col in [:homo, :lumo, :gap, :cv, :zpve]
        y .*= HARTREE_TO_EV
    end
    return y
end

function figure_permutations(name_df::Pair...; name_df_order)
    f = Figure(size=(3.42inch, 2inch), figure_padding=(2, 3, 2, 4))
    gl_trends = GridLayout(f[1, 1])
    axes = [
        :homo => L"HOMO$$",
        :gap => L"Gap$$",
        :lumo => L"LUMO$$",
        :zpve => L"ZPVE$$",
        # :cv => L"CV$$",
        :h298 => L"$G\degree$",
    ]
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
            yticks=LogTicks(WilkinsonTicks(5)),
            yminorticksvisible=true,
            xminorticks=IntervalsBetween(5),
        )
        ax_scatter = Axis(gl_trends[idx, 2];
            xscale=ax_tend.yscale,
            yscale=ax_tend.yscale,
            xlabel="Randomized\nUncertainty",
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
        if col in [:homo, :lumo, :gap, :cv, :zpve]
            y .*= HARTREE_TO_EV
        end
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
        linkyaxes!(ax, axs)
        powerlaw!(axs, 1, 1; color=:black)
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

    # Order Sensitivity
    n_carbon_range = extrema(last(first(name_df_order)).n_carbon)
    gl_order = GridLayout(f[1, 2])
    cb = Colorbar(gl_order[1:length(name_df_order), 2];
        label="Number of Carbons",
        colorrange=n_carbon_range,
        width=5
    )
    axes = Axis[]
    for (idx, (_, df)) in enumerate(name_df_order)
        is_last = idx == length(name_df_order)
        ax = Axis(gl_order[idx, 1];
            xlabel=L"Double Bond Location$$",
            ylabel=L"HOMO [eV]$$",
            limits=((0, 1), nothing),
            xtickformat="{:.0%}",
            xlabelvisible=is_last,
            xticksvisible=is_last,
            xticklabelsvisible=is_last,
        )
        push!(axes, ax)
        df = subset(df, :n_carbon => ByRow(>(4)))
        # df = subset(df, :n_carbon => ByRow(n -> n % 2 == 0))
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
    linkyaxes!(axes...)
    colgap!(gl_order, 1, 4)
    colgap!(f.layout, 1, 3)
    colsize!(f.layout, 1, Relative(3 / 4))

    label_kwargs = (;
        fontsize=8pt,
        font=:bold,
        halign=:right,
        tellheight=false,
    )
    Label(f[1, 1, TopLeft()], "a)"; padding=(0, 15, 0, 2), label_kwargs...)
    Label(gl_order[1, 1, TopLeft()], "b)"; padding=(0, 7, 0, 2), label_kwargs...)
    Label(gl_order[2, 1, TopLeft()], "c)"; padding=(0, 7, 0, 7), label_kwargs...)

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
