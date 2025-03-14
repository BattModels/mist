function hydrocarbon_trends(df)
    f = Figure()

    # Trends with Size
    df = rename(df, "BF3 affinity" => :dn)
    gl_trends = GridLayout(f[1, 1])
    rowgap!(gl_trends, 5)
    axes = [
        :mu => L"$\mu$ [D]",
        :h298 => L"$G\degree$\n[kJ/mol]",
        :alpha => L"$\alpha$\n[$\alpha_0^2$]",
        :mp => L"$$Melt\n[$\degree C$]",
        :dn => L"$$DN\n[kJ/mol]",
        :pKa_kt => L"pKa",
        :alpha_kt => L"KT $\alpha$",
        :beta_kt => L"KT $\beta$",
    ]
    axes = map(enumerate(axes)) do (idx, (col, ylabel))
        is_last = idx == length(axes)
        col => Axis(gl_trends[idx, 1];
            xlabel=L"$$Number of Carbons", ylabel,
            limits=((1, 25), nothing),
            xlabelvisible=is_last,
            xticksvisible=is_last,
            xticklabelsvisible=is_last,
        )
    end |> Dict
    foreach(groupby(df, :type)) do gdf
        for (col, ax) in pairs(axes)
            errorlines!(ax, gdf.n_carbon, gdf[:, col]; label=first(gdf.type))
        end
    end


    # Design Rules
    gl_dr = GridLayout(f[1, 2])
    ax = Axis(gl_dr[1, 1];
        xlabel="HOMO [eV]", ylabel="pKa",
    )
    cb = Colorbar(gl_dr[1, 2];
        label="DN",
        colorrange=(0, 30),
    )
    errorcross!(ax, df.homo .* 27.2114, df.pKa_kt;
        # color=mean.(df.dn),
        color=df.n_carbon,
        cb_attrs(cb)...,
    )

    rowgap!(gl_trends, 5)
    resize_to_layout!(f)

    return f
end
