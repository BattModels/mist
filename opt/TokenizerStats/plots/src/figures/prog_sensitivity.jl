function figure_prog_sense(fe_models, prog_models; scale=:relative, colormap=:tab10, positive_is_better=true)
    f = Figure(size=(7inch, 2.5inch))

    datasets = OrderedDict(
        "tmQM" => "tmQM",
        "qm9" => "QM9",
        "qm8" => "QM8",
        "lipo" => "Lipo.",
        "hiv" => "HIV",
        "toxcast" => "ToxCast",
        "tox21" => "Tox21",
        "clintox" => "ClinTox",
        "freesolv" => "FreeSolv",
        "esol" => "ESOL",
        "bbbp" => "BBBP",
    )
    coefnames = OrderedDict(
        "(Intercept)" => missing,
        "ng_loss_avg" => "N-Gram Loss (Pretrained)",
        "ng_ft_loss_avg" => "N-Gram Loss (Finetuned)",
        "info_loss_avg" => "Info. Loss",
    )
    ds_metric = Dict(eachrow(subset(fe_models, :ngram => ByRow(!))[!, [:dataset, :metric]]))

    # Block out figure
    n_datasets = length(datasets)
    gl_coef = GridLayout(f[1, 1])
    label_kwargs = (;
        fontsize=10pt, font=:bold,
        halign=:right,
        tellheight=false,
    )
    for (adx, ds) in enumerate(keys(datasets))
        plt_ds = datasets[ds]
        metric = uppercase(ds_metric[ds])
        ds_models = subset(prog_models, :dataset => ByRow(==(ds)), :finetuned)
        @assert nrow(ds_models) >= 1
        ds_models.metric = map(ds -> ds_metric[ds], ds_models.dataset)
        better_sign =  map(ds_models.metric) do metric
            metric in ["auroc"] ? 1 : -1
        end

        val, ci, kwargs = val_ci_dodge(ds_models.loss_and_info_and_ft;
            scale,
            better_sign = positive_is_better ? better_sign : nothing,
            coefs=keys(coefnames),
        )

        # Plot Effect sizes
        lb, ub = round.(kwargs.baseline.credible_interval; sigdigits=3)
        if lb != ub
            title = "$plt_ds\n$metric: $lb - $ub"
        else
            title = "$plt_ds\n$metric: $lb"
        end
        yticks = (first(kwargs.yticks), map(k -> get(coefnames, k, k), last(kwargs.yticks)))
        ax = Axis(gl_coef[1, adx];
            title,
            yticks=yticks,
            yticklabelsvisible = adx == 1,
            yticksvisible=adx == 1,
            xlabel=kwargs.effectlabel,
            xticks=WilkinsonTicks(3),
            xminorticksvisible=true,
            xtickformat=kwargs.effectformat,
        )
        effectbars!(ax, val, ci; dodge=kwargs.dodge, color=kwargs.dodge, colormap)

        # Plot Predictions
        row = subset(prog_models,
            :dataset => ByRow(==(ds)),
            :finetuned => ByRow(==(true)),
        ) |> only
        qm = row.loss_and_info_and_ft
        spearman = SpearmanTTest(qm)
        rho = round(spearman.rho; sigdigits=3)
        qm_r2 = round(r2(qm); sigdigits=3)
        # ax = Axis(gl_parity[1, adx-1];
        #     title = L"%$plt_ds, $R^2: %$qm_r2$ $\rho: %$rho$",
        #     ylabel="Transformer - $metric",
        #     xlabel="N-Gram Based Estimate $metric",
        #     xticks=WilkinsonTicks(3),
        #     yticks=WilkinsonTicks(3),
        # )
        # ablines!(ax, 0, 1; color=:black, linestyle=:dash, label="Parity")
        # scatter!(ax, predict(qm), response(qm))
    end

    return f
end

function table_prog_sense(fe_models, prog_models; fig_dir=joinpath(pkgdir(TokenizerStats), "fig"))
    datasets = OrderedDict(
        "tmQM" => "tmQM",
        "qm9" => "QM9",
        "qm8" => "QM8",
        "lipo" => "Lipo.",
        "hiv" => "HIV",
        "toxcast" => "ToxCast",
        "tox21" => "Tox21",
        "clintox" => "ClinTox",
        "freesolv" => "FreeSolv",
        "esol" => "ESOL",
        "bbbp" => "BBBP",
        "bace" => "BACE",
        "sider" => "SIDER",
    )
    coefnames = Dict(
        "ng_loss_avg" => "N-Gram Loss (Pretrained)",
        "ng_ft_loss_avg" => "N-Gram Loss (Finetuned)",
        "info_loss_avg" => "Info. Loss",
    )
    ds_metric = Dict(eachrow(subset(fe_models, :ngram => ByRow(!))[!, [:dataset, :metric]]))
    prog_models = transform(prog_models, :dataset => ByRow(ds -> ds_metric[ds]) => :metric)
    subset!(prog_models, :finetuned)

    # Regression
    lm_reg = subset(prog_models, :metric => ByRow(!=("auroc")))
    lm_class = subset(prog_models, :metric => ByRow(==("auroc")))
    for (file, df) in [("prog_sense_reg.tex", lm_reg), ("prog_sense_class.tex", lm_class)]
        regtable(
            df.loss_and_info_and_ft...;
            below_statistic = RegressionTables.ConfInt,
            labels=coefnames,
            groups=map(ds -> datasets[ds], df.dataset),
            regression_statistics=[
                RegressionTables.R2,
            ],
            render = LatexTable(),
            file = joinpath(fig_dir, file),
        )
        @info joinpath(fig_dir, file)
    end
    return nothing
end
