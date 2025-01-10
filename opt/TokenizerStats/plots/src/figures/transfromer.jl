function figure_ngram_vs_transformer(loss_stats, dfp, dff)
    tokenizers = tokenizers_info()
    val_loss = subset(loss_stats,
        :split => ByRow(==("val")),
        :dataset => ByRow(==("realspace")),
        :tokenizer => ByRow(x -> haskey(tokenizers, x)),
    )
    best_models = combine(groupby(val_loss, :tokenizer)) do gdf
        sort!(gdf, :avg_model_loss; rev=false)
        return gdf[1, :]
    end
    select!(best_models, :tokenizer, :ngram, :avg_model_loss => :ngram_val_loss, :avg_model_token_loss => :ngram_val_token_loss)

    f = Figure(; size=(3.25inch, 1.5inch), figure_padding=(1, 1, 1, 4))

    # Get transformer model pretraining loss
    dfp = leftjoin(dfp, best_models, on=[:tokenizer])
    sort!(dfp, :val_loss)
    dfp.tokenizer = categorical(dfp.tokenizer, levels=unique(dfp.tokenizer))
    replace!(dfp.encoding,
        "smiles" => "SMILES",
        "smiles-canonical" => "Canonical SMILES",
        "selfies" => "SELFIES"
    )
    dfp.encoding = categorical(dfp.encoding, levels=["SMILES", "Canonical SMILES", "SELFIES"])

    toks = levels(dfp.tokenizer)
    ax = Axis(f[1, 1];
        limits=(nothing, (0, nothing)),
        ylabel="Transformer [nats/token]",
        xticklabelrotation=0.4,
        xticks=(1:length(toks), map(n -> tokenizers[n]["name"], toks)),
    )
    h = barplot!(ax, levelcode.(dfp.tokenizer), dfp.val_loss;
        dodge=replace(levelcode.(dfp.encoding), 3 => 2),
        color=levelcode.(dfp.encoding),
        colormap=:Set1_3,
        colorrange=(1, length(levels(dfp.encoding))),
    )

    ds_elements = map(enumerate(levels(dfp.encoding))) do (gdx, encoding)
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
    ols = glm(@formula(val_loss ~ 1 + log(ngram_val_token_loss)), dfp, Normal(), LogLink())
    display(ols)

    ax = Axis(f[1, 2];
        xlabel="n-gram [nats/token]",
        ylabel="Transformer [nats/token]",
        xscale=log10,
        yscale=log10,
        limits=((1, 3), (2e-2, 1)),
    )
    powerlaw!(ax, exp(coef(ols)[1]), coef(ols)[2]; linewidth=0.5, color=:black)

    scatter!(ax, dfp.ngram_val_token_loss, dfp.val_loss;
        color=levelcode.(dfp.encoding),
        colormap=h.colormap,
        colorrange=h.colorrange,
    )

    colsize!(f.layout, 1, Relative(2 / 3))
    colgap!(f.layout, 5)
    resize_to_layout!(f)

    return f
end


function figure_tf_finetune(dff, dft, stats_dir)
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
        TokenizerStats.dodgederrorbars!(ax, levelcode.(x), y, std;
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

