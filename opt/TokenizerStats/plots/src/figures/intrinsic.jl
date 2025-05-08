function figure_intrinsic(df)
    f = Figure(; size=(3.42inch, 1.5inch))
    colgap!(f.layout, 3pt)
    tokenizer_classes = OrderedDict(
        "bpe, nlp" => "BPE, NLP",
        "bpe" => "BPE, Chemistry",
        "atomwise" => "Atom-wise",
        "spe" => "SPE/APE",
        "character" => "Character",
        "unigram" => "Unigram",
        "smirk-gpe" => "Smirk-GPE",
        "smirk" => "Smirk",
    )
    df = subset(df, :tokenizer_domain => ByRow(in(["nlp", "chemistry"])))
    df.tokenizer_class = map(df.tokenizer_domain, df.tokenizer_class) do domain, tokenizer_class
        if domain == "chemistry"
            return tokenizer_class
        else
            return "$tokenizer_class, $domain"
        end
    end
    df.tokenizer_class = categorical(df.tokenizer_class, levels=(collect∘keys)(tokenizer_classes), ordered=true)

    df.dataset = map(ds -> ds in ["realspace", "tmQM"] ? ds : "MoleculeNet", df.dataset)
    df.dataset = map(ds -> ds == "realspace" ? "REALSpace" : ds, df.dataset)
    df.dataset = categorical(df.dataset, levels=["REALSpace", "tmQM", "MoleculeNet"], ordered=true)
    colormap=:Set1_3
    colorrange=(1, 3)

    metrics = [:fertility, :divergence, :normalized_entropy, :oov_rate]
    ax_kwargs = Dict(
        :fertility => (;
            title="Fertility",
            limits=((0, nothing), nothing),
        ),
        :divergence => (;
            title="Imbalance",
            limits=((0.4, nothing), nothing),
            xtickformat="{:.0%}",
        ),
        :oov_rate => (;
            title="UNK Freq.",
            limits=((0, 1), nothing),
            xtickformat="{:.0%}",
            xscale=my_sqrt,
        ),
        :normalized_entropy => (;
            title=L"\eta",
            limits=((0, 1), nothing),
            xticks=[0.25, 0.75],
            xtickformat="{:.0%}",
        ),
    )
    # ax_kwargs = Dict()

    for (mdx, metric) in enumerate(metrics)
        kwargs = get(ax_kwargs, metric, (;))
        ax = Axis(f[1, mdx];
            yticks=(1:length(tokenizer_classes), collect(values(tokenizer_classes))),
            yticksvisible=false,
            yticklabelsvisible=mdx == 1,
            xticklabelrotation=-pi/4,
            xminorticks=IntervalsBetween(4),
            xminorticksvisible=true,
            kwargs...
        )
        for (gdx, gdf) in enumerate(keys(tokenizer_classes))
            gdf = subset(df, :tokenizer_class => ByRow(==(gdf)))
            x = gdx * ones(nrow(gdf))
            y = collect(gdf[!, metric])
            h = boxplot!(ax, x, y;
                orientation=:horizontal,
                dodge=levelcode.(gdf.dataset),
                color=levelcode.(gdf.dataset),
                show_outliers=false,
                whiskerwidth=1pt,
                colormap,
                colorrange,
            )
        end
    end
    ds_elements = map(enumerate(levels(df.dataset))) do (color, label)
        PolyElement(; label, color, colorrange, colormap)
    end
    Legend(f[1, end], ds_elements, labels(ds_elements);
        tellheight=false, tellwidth=false, orientation=:vertical,
        framevisible=true,
        margin=(2, 2, 2, 2),
        fontsize=6pt,
        patchsize=(6pt, 6pt),
        padding=2pt,
        rowgap=1pt,
        valign=:top,
        halign=:right,
    )
    resize_to_layout!(f)

    return f
end
