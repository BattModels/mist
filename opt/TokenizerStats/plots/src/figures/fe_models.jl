function figure_fe_models(models, prog_models; scale=:relative, colormap=:tab10, positive_is_better=true)
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
    ds_metric = Dict(eachrow(subset(models, :ngram => ByRow(!))[!, [:dataset, :metric]]))

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

    # Block out figure
    n_levels = length(unique(models.label))
    n_datasets = length(datasets)
    colorrange = (1, n_levels)
    gl_coef = GridLayout(f[1, 1]) #range(1; length=n_datasets)])
    gl_prog_effect = GridLayout(f[2, 1])
    gl_parity = GridLayout(f[2, 1]) #range(1; length=n_datasets-1)])
    label_kwargs = (;
        fontsize=10pt, font=:bold,
        halign=:right,
        tellheight=false,
    )
    Label(f[1, 1, TopLeft()], "a)"; padding=(0, 15, 3, 0), label_kwargs...)
    Label(f[2, 1, TopLeft()], "b)"; padding=(0, 15, 3, 0), label_kwargs...)
    # Label(f[2, 2, TopLeft()], "c)"; padding=(0, 15, 5, 0), label_kwargs...)

    for (adx, ds) in enumerate(keys(datasets))
        plt_ds = datasets[ds]
        ds_models = subset(models, :dataset => ByRow(==(ds)))
        metric = uppercase(ds_metric[ds])
        better_sign =  map(ds_models.metric) do metric
            metric in ["auroc"] ? 1 : -1
        end

        val, ci, kwargs = val_ci_dodge(ds_models.model;
            scale,
            better_sign = positive_is_better ? better_sign : nothing,
            dodge=levelcode.(ds_models.label),
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
        effectbars!(ax, val, ci; dodge=kwargs.dodge, color=kwargs.dodge, colormap, colorrange)

        if adx == 1
            h = map(enumerate(levels(models.label))) do (color, label)
                PolyElement(; label, color, colormap, colorrange)
            end
            Legend(gl_coef[1, 1], h, labels(h);
                tellheight=false,
                tellwidth=false,
                valign=:bottom,
                halign=positive_is_better ? :left : :right,
                margin=(2, 2, 2, 2),
                alignmode=Outside(),
                labelsize=6pt,
                patchsize=(6pt, 6pt),
            )
        end

        # Plot Predictions
        if adx != 1
            row = subset(prog_models,
                :dataset => ByRow(==(ds)),
                :finetuned => ByRow(==(true)),
            ) |> only
            qm = row.loss_and_info_and_ft
            spearman = SpearmanTTest(qm)
            rho = round(spearman.rho; sigdigits=3)
            qm_r2 = round(r2(qm); sigdigits=3)
            ax = Axis(gl_parity[1, adx-1];
                title = L"%$plt_ds, $R^2: %$qm_r2$ $\rho: %$rho$",
                ylabel="Transformer - $metric",
                xlabel="N-Gram Based Estimate $metric",
                xticks=WilkinsonTicks(3),
                yticks=WilkinsonTicks(3),
            )
            ablines!(ax, 0, 1; color=:black, linestyle=:dash, label="Parity")
            scatter!(ax, predict(qm), response(qm))
        end
    end

    # # Plot Prog Effect Size
    # prog_effect = stack(coefint, prog_models.loss_and_info_and_ft)
    # val, ci, kwargs = val_ci_dodge(prog_models.loss_and_info_and_ft;
    #     scale=:std2,
    #     better_sign=map(ds -> ds in ["tmQM", "sider", "QM9", "lipo"] ? -1 : 1, prog_models.dataset)
    # )
    # ax = Axis(gl_prog_effect[1,1];
    #     yticks=kwargs.yticks,
    #     limits=(nothing, (0, 4)),
    # )
    # effectbars!(ax, val, ci;
    #     dodge=kwargs.dodge,
    #     color=kwargs.dodge,
    #     colormap,
    #     colorrange=(1, size(prog_effect, 2))
    # )

    return f
end

function val_ci_dodge(models; dodge::Union{AbstractVector,Nothing}=nothing, scale=:absolute, coefs=Colon(), better_sign=nothing)
    names = coefs isa Colon ? union(coefnames.(models)...) : collect(coefs)
    ci = stack(m -> coefint(m, names), models)
    baseline = ci[1]

    # Rescale coefficients
    effectlabel = "Effect Size"
    effectformat = "{:.2f}"
    if scale == :relative
        ci = ci ./ ci[[1], :]
        effectlabel = "Relative Effect Size"
        effectformat = "{:.0%}"
    elseif scale == :std
        ci = ci ./ map(std∘response, models)'
        effectlabel = "Effect Size (Std. Dev.)"
    elseif scale == :std2
        response_std = map(std∘response, models)'
        coef_std = stack(m -> vec(std(modelmatrix(m); dims=1)), models)
        ci = ci .* (coef_std ./ response_std)
    elseif scale == :coef_scale
        ci = ci ./ span.(ci[1, :])'
    elseif scale != :absolute
        error("unknown scale $scale")
    end

    # Drop intercept
    ci = ci[2:end, :]
    names = names[2:end]

    # Flip so positive is better
    if !isnothing(better_sign)
        ci = ci .* vec(better_sign)'
        effectlabel = "Pos. " * effectlabel
    end

    # Pack results
    val = vec(repeat(1:size(ci, 1), 1, size(ci, 2)))
    if isnothing(dodge)
        dodge = 1:size(ci, 2)
    end
    dodge = vec(repeat(dodge', size(ci, 1), 1))
    ci = vec(ci)
    val = val[(!ismissing).(ci)]
    dodge = dodge[(!ismissing).(ci)]
    ci = ci[(!ismissing).(ci)]

    kwargs = (;
        dodge,
        effectlabel,
        effectformat,
        yticks = (1:length(names), names),
        baseline,
    )
    return val, ci, kwargs
end
