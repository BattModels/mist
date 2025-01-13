
function figure_ngram_vs_transformer(stats_dir, df)

    # Get transformer model pretraining loss
    tokenizers = tokenizers_info(stats_dir)

    f = Figure(; size=(3.25inch, 1.5inch), figure_padding=(10, 1, 1, 4))
    xticks = map(enumerate(levels(df.tokenizer))) do (id, tok)
        name = tokenizers[tok]["name"]
        class = tokenizers[tok]["tokenizer_class"]
        cls_label = get(CLASS_PLT_LABEL, class, class)
        return (id, name)
        # return (id, lowercase(name) != class ? "$name - $cls_label" : name)
    end
    xticks = (first.(xticks), last.(xticks))

    ax = Axis(f[1, 1];
        limits=(nothing, (0, nothing)),
        ylabel="Transformer [nats/token]",
        xticklabelrotation=0.4,
        xticks,
    )
    h = barplot!(ax, levelcode.(df.tokenizer), df.val_loss;
        dodge=replace(levelcode.(df.encoding), 3 => 2),
        color=levelcode.(df.encoding),
        colormap=:Set1_3,
        colorrange=(1, length(levels(df.encoding))),
    )

    ds_elements = map(enumerate(levels(df.encoding))) do (gdx, encoding)
        PolyElement(label=encoding, color=gdx, colorrange=h.colorrange, colormap=h.colormap)
    end
    Legend(f[1, 1], ds_elements, map(e -> e.label, ds_elements);
        tellheight=false, tellwidth=false, orientation=:vertical,
        framevisible=true,
        patchstrokecolor=:black,
        patchstrokewidth=1,
        margin=(2, 2, 2, 2),
        padding=3,
        valign=:top,
        halign=:left,
    )


    # Test if N-Gram's predict pretraining loss
    ols = glm(@formula(val_loss ~ 1 + log(ngram_token_loss)), df, Normal(), LogLink())
    display(ols)

    ax = Axis(f[1, 2];
        xlabel="n-gram [nats/token]",
        ylabel="Transformer [nats/token]",
        xscale=log10,
        yscale=log10,
        limits=((1, 3), (2e-2, 1)),
    )
    powerlaw!(ax, exp(coef(ols)[1]), coef(ols)[2]; linewidth=0.5, color=:black)

    scatter!(ax, df.ngram_token_loss, df.val_loss;
        color=levelcode.(df.encoding),
        colormap=h.colormap,
        colorrange=h.colorrange,
    )

    label_kwargs = (;
        fontsize=12pt, font=:bold,
        halign=:right,
        tellheight=false,
    )
    Label(f[1, 1, TopLeft()], "A)"; padding=(0, 20, -5, 0), label_kwargs...)
    Label(f[1, 2, TopLeft()], "B)"; padding=(0, 23, -5, 0), label_kwargs...)
    colsize!(f.layout, 1, Relative(2 / 3))
    colgap!(f.layout, 5)
    resize_to_layout!(f)

    return f
end


function figure_tf_finetune(stats_dir, dff, dft)
    f = Figure(; size=(5inch, 2.5inch), figure_padding=(1, 1, 1, 4))
    tokenizers = tokenizers_info(stats_dir)
    dff.tokenizer = categorical(dff.tokenizer)
    dff.dataset = categorical(dff.dataset)

    # Tabulated Results for SI
    colormap = distinguishable_colors(length(levels(dff.tokenizer)), [colorant"white", colorant"black"];
        dropseed=true,
        lchoices=range(0, stop=70, length=15), # Avoid light colors
    )

    # Get test results
    dft = subset(dft, :channel => ByRow(isnothing), :tok_group => ByRow(==("all")))
    select!(dft, Not(:tok_group, :channel))
    rename!(dft, :mean => :test_loss, :std => :test_loss_std, :id => :test_id)
    dff = leftjoin(dff, dft; on=[:id => :ckpt_id, :metric])

    # Merge smiles/canonical/kekule encoding into one bar (pick best by val loss)
    dff_all = dff # Save for reporting
    select!(dff, Not(:train_oov_loss, :val_oov_loss))
    dff = combine(groupby(dff, [:tokenizer, :dataset])) do gdf
        ids = argmin(gdf.val_loss)
        return gdf[ids, :]
    end

    # Plot Regression
    dfr = subset(dff, :task => ByRow(==("regression")))
    dfr.dataset = categorical(string.(dfr.dataset))
    ax = Axis(f[1, 1];
        limits=(nothing, (0, 1)),
        ylabel="Test R2",
        xticks=categorical_ticks(dfr.dataset),
        xticklabelrotation=0.4,
        yticks=LinearTicks(5),
        ytickformat="{:.0%}",
    )
    h = _finetune_results!(ax, dfr.dataset, dfr.test_loss, dfr.tokenizer;
        std=dfr.test_loss_std,
        colormap,
        colorrange=(1, length(colormap)),
    )

    # Skip MUV as it use AU-PRC, not AUROC
    dfc = subset(dff, :task => ByRow(==("binary")), :dataset => ByRow(!=("muv")))
    dfc.dataset = categorical(string.(dfc.dataset))
    ax = Axis(f[2, 1];
        limits=(nothing, (0, 1)),
        ylabel="Test AUROC",
        xticks=categorical_ticks(dfc.dataset),
        xticklabelrotation=0.4,
        yticks=LinearTicks(5),
        ytickformat="{:.0%}",
    )
    _finetune_results!(ax, dfc.dataset, dfc.test_loss, dfc.tokenizer;
        std=dfc.test_loss_std,
        colormap=h.colormap,
        colorrange=h.colorrange
    )

    # Dataset Legend
    ds_levels = levels(dff.tokenizer)
    ds_elements = map(1:length(ds_levels)) do gdx
        PolyElement(polycolor=gdx, colormap=h.colormap, colorrange=h.colorrange)
    end
    Legend(f[1:2, 2], ds_elements, map(tok -> tokenizers[tok]["name"], ds_levels);
        tellheight=false, tellwidth=true, orientation=:vertical,
        patchstrokecolor=:black,
        framevisible=false,
    )

    resize_to_layout!(f)
    return f, dff_all
end

function _finetune_results!(ax, x, y, dodge; std=nothing, colormap, colorrange=nothing)
    if isnothing(colorrange)
        colorrange = extrema(levelcode.(dodge))
    end

    h = barplot!(ax, levelcode.(x), y;
        dodge=levelcode.(dodge),
        colormap,
        colorrange,
        color=levelcode.(dodge),
    )

    if !isnothing(std)
        SmirkPaperPlots.dodgederrorbars!(ax, levelcode.(x), y, std;
            dodge=h.dodge,
            width=h.width,
            n_dodge=h.n_dodge,
            gap=h.gap,
            dodge_gap=h.dodge_gap,
            linewidth=1,
            color=:black,
        )
    end

    return h
end

categorical_ticks(x) = (1:length(levels(x)), levels(x))

function table_finetuned_models(stats_dir, dff, dft)
    dff = select(dff, [:id, :pretrained_id, :tokenizer, :task, :dataset, :encoding])
    dff.benchmark = map(dff.dataset, dff.task) do dataset, task
        if task == "regression"
            if dataset in ["qm8", "qm9", "tmQM"]
                return "mae"
            elseif dataset in ["esol", "freesolv", "lipo"]
                return "rmse"
            end
        else
            return "auroc"
        end
    end
    tokenizers = tokenizers_info(stats_dir)
    dft = subset(dft, :tok_group => ByRow(==("all")))
    select!(dft, Not(:tok_group))
    dfb = innerjoin(dft, dff, on=[:ckpt_id => :id, :metric => :benchmark])
    dfb.tokenizer_name = map(tok -> tokenizers[tok]["name"], dfb.tokenizer)
    dfb = subset(dfb, :encoding => ByRow(!=("smiles-canonical")))
    dfb = combine(groupby(dfb, [:dataset, :channel])) do gdf
        score = collect(gdf.mean)
        gdf.rank = sortperm(score; rev=first(gdf.metric) ∉ ["mae", "rmse"]) |> invperm
        return gdf
    end


    dfb.value = map(dfb.dataset, dfb.metric, dfb.mean, dfb.std, dfb.rank) do dataset, task, mean, std, rank
        if dataset in ["qm8"]
            fmt = "{:.4f} \\pm {:.4f}"
        else
            fmt = "{:.3f} \\pm {:.3f}"
        end
        if task == "auroc"
            mean *= 100
            std *= 100
        end
        s = format(fmt, mean, std)
        if rank == 1
            s = "\\mathbf{$s}"
        end
        return "\\($s\\)"
    end


    subset!(dfb, :channel => ByRow(isnothing))
    tab = unstack(dfb, [:tokenizer, :tokenizer_name], :dataset, :value)
    display(tab)

    # Regression
    println("\nRegression:\n")
    select(tab, [:tokenizer_name, :qm8, :qm9, :tmQM, :esol, :freesolv, :lipo]) |> latex

    # Classification
    println("\nClassification:\n")
    select(tab, [:tokenizer_name, :hiv, :bace, :clintox, :tox21, :toxcast, :sider]) |> latex




    return dfb, tab
end

function anova_explanatory(mu, std, n)
    m = sum(mu .* n) / sum(n)
    SSt = @. n * (mu - m)^2
    SSe = @. (n - 1) * std^2
    J = length(mu)
    t = ("One-way analysis of variance (ANOVA) test", "Means", "F")
    return VarianceEqualityTest{FDist}(n, SSt, SSe, J - 1, sum(n) - J, t)
end

latex(df::DataFrame) = latex(stdout, df)
function latex(io::IO, df::DataFrame)
    println(io, join(names(df), " & ") * " \\\\")
    for row in eachrow(df)
        println(io, join(values(row), " & ") * " \\\\")
    end
end
