function figure_fe_models(models, prog_models; relative=true, colormap=:tab10)
    f = Figure(size=(7inch, 2.5inch))

    datasets = OrderedDict(
        "realspace" => "REALSpace",
        "tmQM" => "tmQM",
        "qm9" => "QM9",
        # "qm8" => "QM8",
        "lipo" => "Lipo.",
        "hiv" => "HIV",
        "toxcast" => "ToxCast",
        "clintox" => "ClinTox",
        # "freesolv" => "FreeSolv",
        # "esol" => "ESOL",
    )
    coefnames = OrderedDict(
        "(Intercept)" => missing,
        "tokenizer_class: bpe" => "BPE",
        "tokenizer_class: smirk" => "Smirk",
        "tokenizer_class: smirk-gpe" => "Smirk-GPE",
        "tokenizer_class: spe" => "SPE/APE",
        "encoding: selfies" => "SELFIES",
    )
    @show ds_metric = Dict(eachrow(subset(models, :ngram => ByRow(!))[!, [:dataset, :metric]]))

    function label_model(ngram, finetuned)
        if !ngram
            return "Transformer"
        elseif ismissing(finetuned) || !finetuned
            return "N-Gram"
        else
            return "N-Gram, Finetuned"
        end
    end
    models = transform(models,
        [:ngram, :finetuned] => ByRow(label_model) => :label
    )
    models.label = categorical(models.label,
        levels=["Transformer", "N-Gram", "N-Gram, Finetuned"],
        ordered=true,
    )
    n_levels = length(unique(models.label))
    colorrange = (1, n_levels)
    gl_coef = GridLayout(f[1, 1])
    gl_parity = GridLayout(f[2, 1])

    for (adx, ds) in enumerate(keys(datasets))
        plt_ds = datasets[ds]
        ds_models = subset(models, :dataset => ByRow(==(ds)))
        metric = uppercase(ds_metric[ds])
        ci = hcat(map(m -> coefint(m, collect(keys(coefnames))), ds_models.model)...)
        yticklabels = collect(values(coefnames))
        baseline = ci[1]
        if relative
            ci = ci ./ ci[[1], :]
        end

        # Drop intercept
        ci = ci[2:end, :]
        yticklabels = yticklabels[2:end]

        # # Flip so positive is better
        # ci_sign = map(ds_models.metric) do metric
        #     metric in ["auroc"] ? 1 : -1
        # end
        # ci = ci .* ci_sign'

        # Plot Effect sizes
        lb, ub = round.(baseline.credible_interval; sigdigits=3)
        if lb != ub
            title = "$plt_ds\n$metric: $lb - $ub"
        else
            title = "$plt_ds\n$metric: $lb"
        end
        ax = Axis(gl_coef[1, adx];
            title,
            yticks=(eachindex(yticklabels), yticklabels),
            yticklabelsvisible = adx == 1,
            yticksvisible=adx == 1,
            xlabel="Relative Effect Size",
            xticks=WilkinsonTicks(3),
            xminorticksvisible=true,
            xtickformat="{:.0%}",
        )
        val = vec(repeat(1:size(ci, 1), 1, size(ci, 2)))
        dodge = levelcode.(ds_models.label)
        dodge = vec(repeat(dodge', size(ci, 1), 1))
        ci = vec(ci)
        val = val[(!ismissing).(ci)]
        dodge = dodge[(!ismissing).(ci)]
        ci = ci[(!ismissing).(ci)]
        effectbars!(ax, val, ci; dodge, color=dodge, colormap, colorrange)

        if adx == 1
            h = map(enumerate(levels(models.label))) do (color, label)
                PolyElement(; label, color, colormap, colorrange)
            end
            Legend(gl_coef[1, 1], h, labels(h);
                tellheight=false,
                tellwidth=false,
                valign=:bottom,
                halign=:right,
                margin=(2, 2, 2, 2),
                alignmode=Outside(),
                labelsize=6pt,
                patchsize=(6pt, 6pt),
            )
        end

        # Plot Predictions
        if adx != 1
            qm = subset(prog_models,
                :dataset => ByRow(==(ds)),
                :finetuned => ByRow(==(true)),
            ) |> only
            qm = qm.loss_and_info_and_ft
            rho = round(corspearman(predict(qm), response(qm)); sigdigits=3)
            qm_r2 = round(r2(qm); sigdigits=3)
            ax = Axis(gl_parity[1, adx-1];
                title = L"%$plt_ds, $R^2: %$qm_r2$ $\rho: %$rho$",
                ylabel="Transformer - $metric",
                xlabel="N-Gram Based Estimate $metric",
            )
            ablines!(ax, 0, 1; color=:black, linestyle=:dash, label="Parity")
            scatter!(ax, predict(qm), response(qm))
        end

    end

    # Add Labels
    label_kwargs = (;
        fontsize=12pt, font=:bold,
        halign=:right,
        tellheight=false,
    )
    Label(f[1, 1, TopLeft()], "A)"; padding=(0, 15, -5, 0), label_kwargs...)
    Label(f[2, 1, TopLeft()], "B)"; padding=(0, 15, -5, 0), label_kwargs...)
    # Label(f[1, 2, TopLeft()], "C)"; padding=(0, 5, -5, 0), label_kwargs...)

    # Label(f[2, :], "Effect Size";
    #     tellwidth=false, tellheight=true,
    #     font=:bold,
    #     fontsize=10pt,
    # )

    # h = map(enumerate(levels(models.label))) do (color, label)
    #     PolyElement(; label, color, colormap, colorrange)
    # end
    # Legend(f[end, 1], h, labels(h);
    #     # orientation=:horizontal,
    #     tellheight=false,
    #     tellwidth=false,
    # )

    return f
end
