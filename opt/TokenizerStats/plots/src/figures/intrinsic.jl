function figure_intrinsic(df; p=90)
    f = Figure(; size=(3.42inch, 1.5inch))
    colgap!(f.layout, 3pt)
    tokenizer_classes = OrderedDict(
        "nlp" => "NLP",
        "character" => "Character",
        "unigram" => "Unigram",
        "bpe" => "BPE",
        "atomwise" => "Atom-wise",
        "spe" => "SPE/APE",
        "smirk-gpe" => "Smirk-GPE",
        "smirk" => "Smirk",
    )
    df = deepcopy(df)
    df.tokenizer_class = map(df.tokenizer_domain, df.tokenizer_class) do domain, tokenizer_class
        if domain == "chemistry"
            return tokenizer_class
        elseif domain in ["nlp", "nlp-science"]
            return "nlp"
        else
            return "$tokenizer_class, $domain"
        end
    end
    ckeys = collect∘keys
    @info unique(df.tokenizer_class)
    subset!(df, :tokenizer_class => ByRow(in(ckeys(tokenizer_classes))))
    df.tokenizer_class = categorical(df.tokenizer_class, levels=ckeys(tokenizer_classes), ordered=true)

    # Label classes with the number of examples
    foreach(eachrow(combine(groupby(df, :tokenizer_class), :tokenizer => length∘unique => :nclass))) do r
        (; tokenizer_class, nclass) = r
        plt_class = tokenizer_classes[tokenizer_class]
        tokenizer_classes[tokenizer_class] = "$plt_class (n=$nclass)"
    end
    @info tokenizer_classes

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
            title=L"Imbalance ($D$)",
            limits=((0.4, nothing), nothing),
            xtickformat="{:.0%}",
        ),
        :oov_rate => (;
            title="UNK Freq.",
            limits=((0, 1), nothing),
            xtickformat="{:.0%}",
            xscale=sqrt,
            xticks=[0.25, 0.5, 1],
        ),
        :normalized_entropy => (;
            title=L"Normalized Entropy ($\eta$)",
            limits=((0, 1), nothing),
            xticks=[0.25, 0.75],
            xtickformat="{:.0%}",
        ),
    )

    for (mdx, metric) in enumerate(metrics)
        kwargs = get(ax_kwargs, metric, (;))
        ax = Axis(f[1, mdx];
            titlefont=:regular,
            yticks=(1:length(tokenizer_classes), collect(values(tokenizer_classes))),
            yticksvisible=false,
            yticklabelsvisible=mdx == 1,
            xticklabelrotation=-pi/4,
            xminorticks=IntervalsBetween(4),
            xminorticksvisible=true,
            kwargs...
        )
        vspan!(ax,
            percentile(df[!, metric], (100 - p) / 2),
            percentile(df[!, metric], (100 + p) / 2),
            color=:black,
            alpha=0.2,
        )
        vlines!(ax, mean(df[!, metric]);
            color=:black,
            linestyle=:dash,
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

    return f
end

function vspan!(ax, lb, ub, n; kwargs...)
    points = Point2f[(lb, 0), (lb, n), (ub, n), (ub, 0)]
    poly!(ax, points; kwargs...)
end

@recipe(VSpan, lb, ub) do scene
    Attributes(;
    )
end

function Makie.plot!(plt::VSpan)
    ax= Makie.current_axis()
    limits = ax.finallimits
    p = lift(limits, plt[:lb], plt[:ub]) do limits, lb, ub
        ly = limits.origin[2]
        uy = limits.origin[2] + limits.widths[2]
        ly, uy = ly < uy ? (ly, uy) : (uy, ly)
        Point2f[(lb, ly), (lb, uy), (ub, uy), (ub, ly)]
    end
    poly!(plt, p; Makie.shared_attributes(plt, Poly)...)
    return plt
end

