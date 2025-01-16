function figure_token_usage(stats_dir)
    f = Figure(;
        size=(6inch, 4.5 * inch),
        figure_padding=(1, 1, 1, 5),
    )
    ax = Axis(f[1, 1];
        limits=((1, 2500), (0, 25)),
        xlabel="Token Rank",
        ylabel="Information Content [nats]",
        xscale=log10,
        xminorticksvisible=true,
        xminorticks=IntervalsBetween(5),
        xminorgridvisible=true,
    )

    rows = []
    tokenizers = tokenizers_info(stats_dir)
    for name_or_path in keys(tokenizers)
        tok_info = tokenizers[name_or_path]
        usage_file = joinpath(stats_dir, tok_info["name_or_path"], "realspace", "usage.jld2")
        isfile(usage_file) || continue

        stats = jldopen(usage_file)
        usage = stats["train"]["ngrams"]["1"]
        usage = Dict{Int,Int}(only(k) => v for (k, v) in pairs(usage))
        vocab_size = stats["tokenizer"][:vocab_size]
        unk_count = pop!(usage, stats["tokenizer"][:unk_token_id], 0)
        close(stats)

        c_token = collate_token_usage(usage, vocab_size; smoothing=0)
        p_token = c_token ./ sum(c_token)

        sort!(p_token, rev=true)
        efficiency = sum(p -> -p * log(p) / log(vocab_size), filter(>(0), p_token))
        H = sum(p -> p > 0 ? -p * log(p) : 0, p_token; init=0.0)
        p_unk = unk_count / sum(c_token)


        class = tok_info["tokenizer_class"]
        encoding = tok_info["encoding"]
        class_label = class * (encoding == "smiles" ? "" : ", $(encoding)")
        class_label = replace(class_label, "smirk" => "ours")
        push!(rows, (;
            name_or_path,
            name=tok_info["name"],
            usage,
            H,
            efficiency,
            vocab_size,
            class,
            class_label,
            p_unk
        ))
    end
    df = DataFrame(rows)
    sort!(df, [order(:efficiency, rev=true), :name, :vocab_size])
    display(select(df, Not(:usage)))

    linestyles = [:dot, :dashdot, :dashdotdot, :solid]
    colors = distinguishable_colors(nrow(df), [colorant"white", colorant"black"];
        dropseed=true,
        lchoices=range(0, stop=70, length=15), # Avoid light colors
    )

    for (idx, row) in enumerate(eachrow(df))
        tokenusage!(ax, row.usage, row.vocab_size;
            smoothing=0,
            label=format("{:.1%} - {:s} ({:s})", row.efficiency, row.name, row.class_label),
            linestyle=linestyles[idx%length(linestyles)+1],
            linewidth=1,
            color=colors[idx],
        )
    end


    Legend(f[1, 2], ax; tellheight=true, tellwidth=true, patchsize=(12, 6), valign=:top)
    colgap!(f.layout, 2)
    resize_to_layout!(f)
    return f
end

function collate_atomic_oov(key::String, results::Dict)
    !haskey(results, key) && return missing, missing
    r = results[key]
    total = r["nobs"] + r["failed_encode"]
    covered = total - r["oov"] - r["failed_encode"]
    return (covered / total), (r["failed_encode"] / total)
end

function collate_atomic_oov(key::Regex, results::Dict)
    nobs = 0
    oov = 0
    failed_encode = 0
    valid = false
    for (k, v) in results
        if !isnothing(match(key, k))
            valid = true
            nobs += v["nobs"] + v["failed_encode"]
            oov += v["oov"]
            failed_encode += v["failed_encode"]
        end
    end
    valid || return missing, missing
    covered = nobs - oov - failed_encode
    return (covered / nobs), (failed_encode / nobs)
end

function figure_oov_rate(stats_dir)
    datasets = OrderedDict(
        "Elements" => "elements",
        "Bonds" => "bonds",
        "Isotopes" => "isotopes",
        "Carbon Rings" => "rings",
        "Ions" => "charged_elements",
        "Chirality" => "chiral_elements",
        "Charged, Chiral Isotopes" => "charged_chiral_isotopes",
        "MoleculeNet" => r"^MoleculeNet/",
        "tmQM" => "tmQM",
    )

    # Tokenizer to include in the figure
    plt_tokenizers = [
        "smirk",
        "smirk-gpe-50k-mb-ss",
        "smirk-selfies",
        "ibm/MoLFormer-XL-both-10pct-oov",
        "ibm/materials.smi-ted-light",
        "seyonec/ChemBERTa-zinc-base-v1",
        "lbnlp/MatBERT-uncased",
        "devalab/molgpt-guacamol",
        "rxn4chemistry/rxnfp",
        "MolecularAI/Chemformer",
        "sagawa/ReactionT5-yield-prediction",
        "SmilesPE/SPE_ChEMBL",
        "mikemayuare/SMILYAPE",
        "mikemayuare/SELFYAPE",
        "HUBioDataLab/SELFormer",
        "ibm/materials.selfies-ted",
        "ChangwenXu98/TransPolymer",
        "ncfrey/ChemGPT-4.7M",
    ]

    tokenizers = tokenizers_info(stats_dir)
    tok_names = String[]
    tok_id = Int[]
    oov_rate = Float64[]
    ds_grp = Int[]
    enc_grp = Int[]
    for (idx, name_or_path) in enumerate(plt_tokenizers)
        tok_info = tokenizers[name_or_path]
        push!(tok_names, tok_info["name"])
        oov_stat_file = joinpath(stats_dir, tok_info["name_or_path"], "oov.json")
        oov_stats = isfile(oov_stat_file) ? JSON.parsefile(oov_stat_file) : Dict()

        # Collate group oov rates
        tok_info["oov_rate"] = Dict{String,Float64}()
        for (gdx, group_key) in enumerate(collect(values(datasets)))
            group_oov_rate, group_encode_rate = collate_atomic_oov(group_key, oov_stats)
            append!(tok_id, (idx, idx))
            append!(oov_rate, (group_oov_rate, group_encode_rate))
            append!(ds_grp, (gdx, gdx))
            append!(enc_grp, (0, 1))
        end
    end

    # Darken encoding failures
    colormap = cgrad(:Set1_9, length(datasets); categorical=true)
    colormap = [c for c in colormap]
    append!(colormap, map(c -> weighted_color_mean(0.2, RGBA(colorant"black"), c), colormap))
    color = ds_grp .+ length(datasets) .* enc_grp

    # Create figure
    f = Figure(;
        size=(7inch, 2inch),
        figure_padding=(5, 1, 1, 5),
    )
    ax = Axis(f[1, 1];
        ylabel="Tokenizer Coverage",
        limits=((0, length(tok_names) + 1), (0, 1)),
        ytickformat="{:.0%}",
        xticklabelrotation=0.4,
        xticks=(1:length(tok_names), tok_names),
        xticklabelsize=10pt,
        yminorticks=IntervalsBetween(5),
        yminorticksvisible=true,
        yminorgridvisible=true,
        xgridvisible=false,
    )
    h = barplot!(ax, tok_id, oov_rate;
        dodge=ds_grp,
        color,
        stack=enc_grp,
        colorrange=(1, length(colormap)),
        colormap,
        strokewidth=0.1,
        strokecolor=:black,
    )
    ds_elements = map(1:length(datasets)) do gdx
        PolyElement(polycolor=gdx, colormap=h.colormap, colorrange=h.colorrange)
    end
    Legend(f[1, 2], ds_elements, collect(keys(datasets));
        tellheight=false, tellwidth=true, orientation=:vertical,
        patchstrokecolor=:black,
        framevisible=false,
    )

    resize_to_layout!(f)
    return f
end

function figure_ngram_fits(df, stats_dir; colormap=:Set2_5)
    plt_toks = [
        # "character",
        "smirk",
        "smirk-gpe-50k-mb-ss",
        "ibm/MoLFormer-XL-both-10pct-oov",
        "ibm/materials.smi-ted-light",
        "devalab/molgpt-moses",
        "rxn4chemistry/rxnfp",
        "SmilesPE/SPE_ChEMBL",
        "MolecularAI/Chemformer",
        "mikemayuare/SMILYAPE",
        "mikemayuare/SELFYAPE",
        "HUBioDataLab/SELFormer",
        "seyonec/ChemBERTa-zinc-base-v1",
        "ncfrey/ChemGPT-4.7M",
        "ChangwenXu98/TransPolymer",
        "sagawa/ReactionT5-product-prediction",
        "facebook/galactica-6.7b",
        "meta-llama/Llama-3.2-1B",
        "Xenova/gpt-4o",
        "google/gemma-7b",
    ]
    df = subset(df,
        :tokenizer => ByRow(x -> x in plt_toks),
        :split => ByRow(==("val")),
    )
    df.tokenizer = categorical(df.tokenizer; levels=plt_toks)
    tokenizers = tokenizers_info(stats_dir)

    # Set up figure
    f = Figure(; size=(7inch, 3inch), figure_padding=(1, 20, 5, 5))
    tokenizer = levels(df.tokenizer)
    plt_label = map(tokenizer) do tok
        name = tokenizers[tok]["name"]
        class = tokenizers[tok]["tokenizer_class"]
        class_label = get(CLASS_PLT_LABEL, class, class)
        return lowercase(name) != class ? "$name - $class_label" : name
    end
    ax_kwargs = (;
        xticks=(1:length(tokenizer), plt_label),
        xticklabelrotation=0.4,
        xticksvisible=false,
        xgridvisible=false,
    )
    ax_pretrain = Axis(f[1, 1];
        ylabel="Enimine REAL Space\nCross Entropy Loss [nats/token]",
        limits=(nothing, (0, 7)),
        ax_kwargs...
    )
    hidexdecorations!(ax_pretrain)
    ax_molnet = Axis(f[2, 1];
        ylabel="MoleculeNet\nCross Entropy Loss [nats/token]",
        limits=(nothing, (0, nothing)),
        ax_kwargs...
    )
    linkxaxes!(ax_pretrain, ax_molnet)
    ax = Axis(f[1:2, 2];
        ylabel="Molecule Net [nats/token]",
        xlabel="Enimine REAL Space\n[nats/token]",
        limits=((0, nothing), (0, nothing)),
    )
    colsize!(f.layout, 2, Fixed(1.5 * inch))
    colgap!(f.layout, 5)

    # N-Gram Legend
    ds_elements = map(1:5) do gdx
        PolyElement(polycolor=gdx;
            colormap,
            colorrange=(1, 5),
        )
    end
    Legend(f[1, 1], ds_elements, ["Unigram", "Bigram", "Trigram", "4-gram", "5-gram"];
        tellheight=false, tellwidth=false, orientation=:horizontal,
        framevisible=true,
        patchstrokecolor=:black,
        patchstrokewidth=1,
        margin=(2, 2, 2, 2),
        padding=3,
        valign=:top,
        halign=:left,
    )

    # Pretraining
    df_pretrain = subset(df,
        :dataset => ByRow(==("realspace")),
    )
    barplot!(ax_pretrain, levelcode.(df_pretrain.tokenizer), df_pretrain.avg_model_token_loss;
        dodge=df_pretrain.ngram,
        color=df_pretrain.ngram,
        colormap,
    )

    # Finetune
    df_finetune = subset(df, :dataset => ByRow(∉(["realspace", "tmQM"])))
    df_finetune = combine(groupby(df_finetune, [:tokenizer, :ngram])) do gdf
        loss_per_token_moments = reduce(merge, gdf.loss_per_token_moments)
        return (;
            avg_model_token_loss=mean(loss_per_token_moments),
            std_model_token_loss=std(loss_per_token_moments),
        )
    end
    barplot!(ax_molnet, levelcode.(df_finetune.tokenizer), df_finetune.avg_model_token_loss;
        dodge=df_finetune.ngram,
        color=df_finetune.ngram,
        colormap,
    )

    # Scatter Plot of Pretrain vs. Finetune 
    df = leftjoin(
        select(subset(df_pretrain, :ngram => ByRow(==(5))), :tokenizer, :avg_model_token_loss => :pretrain),
        select(subset(df_finetune, :ngram => ByRow(==(5))), :tokenizer, :avg_model_token_loss => :finetune);
        on=:tokenizer,
    )
    df.class = map(name -> tokenizers[name]["tokenizer_class"], df.tokenizer) |> categorical
    df.domain = map(name -> tokenizers[name]["domain"], df.tokenizer) |> categorical
    markers = [:x, :+, :diamond, :square]
    h = scatter!(ax, df.pretrain, df.finetune;
        marker=map(i -> markers[levelcode(i)], df.domain),
        colormap=:Dark2_6,
        color=levelcode.(df.class),
        colorrange=(1, length(levels(df.class))),
    )
    domain_labels = Dict(
        "chemistry" => "Chemistry",
        "nlp" => "NLP",
        "nlp-science" => "NLP, Science",
    )
    domains = map(enumerate(levels(df.domain))) do (i, domain)
        label = domain_labels[string(domain)]
        MarkerElement(; label, marker=markers[i], color=:black)
    end
    classes = map(enumerate(levels(df.class))) do (i, class)
        class = string(class)
        label = get(CLASS_PLT_LABEL, class, class)
        PolyElement(;
            label,
            marker=:x,
            color=i,
            colormap=h.colormap,
            colorrange=h.colorrange
        )
    end
    labels(x) = map(e -> e.label[], x)

    Legend(f[2, 1:2],
        [domains, classes],
        [labels(domains), labels(classes)],
        ["Domain", "Tokenizer Class"];
        halign=:right, valign=:bottom,
        nbanks=3,
        titleposition=:top,
        margin=(5, 2, 2, 5),
    )

    # Add Labels
    label_kwargs = (;
        fontsize=12pt, font=:bold,
        halign=:right,
        tellheight=false,
    )
    Label(f[1, 1, TopLeft()], "A)"; padding=(0, 15, -5, 0), label_kwargs...)
    Label(f[2, 1, TopLeft()], "B)"; padding=(0, 15, -5, 0), label_kwargs...)
    Label(f[1, 2, TopLeft()], "C)"; padding=(0, 5, -5, 0), label_kwargs...)

    rowgap!(f.layout, 2)
    resize_to_layout!(f)
    return f
end
