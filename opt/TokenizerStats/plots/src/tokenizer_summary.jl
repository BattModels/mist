function top_k_tokens(ngram_file; k=5)
    # Load unigram statistics
    unigram, tok = jldopen(ngram_file, "r") do data
        # Load tokenizer
        name = data["tokenizer"][:tokenizer_name]
        name = startswith(name, "smirk-gpe") ? "./" * name : name
        tok = load_tokenizer(name)

        # Get unigram stats
        unigram = data["train"]["ngrams"]["1"]
        return unigram, tok
    end

    # Find the top k tokens
    id_count = collect(unigram)
    sort!(id_count; rev=true, by=last)
    top_k = [first(id) for (id, _) in id_count[1:k]]

    # Get the tokens
    return pyconvert(Vector{String}, tok.convert_ids_to_tokens(top_k))
end

function tokenizer_dataset_avg(model_loss; metric=:loss_per_token_moments)
    # Collect Avg. Tokenizer Cross-Entropy By Dataset
    transform!(model_loss, :dataset => ByRow(x -> x ∉ ["tmQM", "realspace"] ? "MoleculeNet" : x) => :dataset)
    model_loss = combine(groupby(model_loss, [:tokenizer, :dataset])) do gdf
        moments = reduce(merge, gdf[!, metric])
        return (;
            :avg => mean(moments),
            :std => std(moments),
        )
    end
    return model_loss
end

""" Compute the top-k tokens and number of carbon tokens for each tokenizer """
function tokenizer_summary(stats_dir; k=5)
    # Tokenizer stats
    rows = []
    for tokenizer in JSON.parsefile(abspath(joinpath(stats_dir, "..", "tokenizers.json")))
        usage_file = joinpath(stats_dir, tokenizer["name_or_path"], "realspace", "usage.jld2")
        if isfile(usage_file)
            vocab_size = jldopen(usage_file) do data
                return data["tokenizer"].vocab_size
            end
            top_tokens = top_k_tokens(usage_file; k)
        else
            vocab_size = missing
            top_tokens = missing
        end

        push!(rows, (;
            tokenizer=tokenizer["name_or_path"],
            domain=tokenizer["domain"],
            class=tokenizer["tokenizer_class"],
            encoding=tokenizer["encoding"],
            vocab_size,
            top_tokens,
        ))
    end
    return DataFrame(rows)
end

function fertility_summary(token_usage; stats_dir="./stats")
    tok_info = SmirkPaperPlots.tokenizers_info(stats_dir)
    token_usage = transform(token_usage,
        :dataset => ByRow(x -> x ∉ ["tmQM", "realspace"] ? "MoleculeNet" : x) => :dataset,
        :tokenizer => ByRow(x -> tok_info[x]["tokenizer_class"]) => :tokenizer_class,
    )
    combine(groupby(token_usage, [:tokenizer_class, :dataset])) do gdf
        dataset = first(gdf.dataset)
        fertility = reduce(merge, gdf.fertility)
        avg_fertility = mean(fertility)
        std_fertility = std(fertility)
        return (;
            dataset=lowercase(dataset) in ("tmqm", "realspace") ? dataset : "MoleculeNet",
            fertility,
            avg_fertility,
            std_fertility,
            fmt_fertility=format("{}\\pm{}", round(avg_fertility; sigdigits=3), round(std_fertility; sigdigits=3)),
        )
    end
end

function report_tokenizer_summary_stats(stats_dir, model_loss, info_loss, usage_stats; k=5)
    fmt(μ, σ) = "\\($(format(round(μ; sigdigits=3))) \\pm $(format(round(σ; sigdigits=3)))\\)"
    tokenizers = JSON.parsefile(abspath(joinpath(stats_dir, "..", "tokenizers.json")))

    model_loss = subset(model_loss, :ngram => ByRow(==(5)), :split => ByRow(==("val")))
    df_loss = tokenizer_dataset_avg(model_loss; metric=:loss_per_token_moments)
    df_loss.cross_entropy = fmt.(df_loss.avg, df_loss.std)
    select!(df_loss, :tokenizer, :dataset, :cross_entropy)

    info_loss = subset(info_loss, :ngram => ByRow(==(5)))
    df_info = tokenizer_dataset_avg(info_loss; metric=:info_loss_moments)
    df_info.info_loss = fmt.(df_info.avg, df_info.std)
    select!(df_info, :tokenizer, :dataset, :info_loss)

    df_fertility = tokenizer_dataset_avg(usage_stats; metric=:fertility)
    df_fertility.fertility = fmt.(df_fertility.avg, df_fertility.std)
    select!(df_fertility, :tokenizer, :dataset, :fertility)

    df = leftjoin(df_loss, df_info; on=[:tokenizer, :dataset])
    leftjoin!(df, df_fertility; on=[:tokenizer, :dataset])
    sort!(df, [:tokenizer, :dataset])

    df_toks = @show tokenizer_summary(stats_dir; k)

    tok_name = map(tokenizers) do tok
        if tok["source"] == "ours"
            if tok["name_or_path"] != "character"
                return "\\gh{BattModels/smirk} $(tok["name"]) (ours)"
            else
                return "\\dd{} $(tok["name"]) (ours)"
            end
        else
            src = tok["source"] == "huggingface" ? "hf" : "gh"
            name_or_path = tok["name_or_path"]
            name_or_path = name_or_path == "SmilesPE/SPE_ChEMBL" ? "XinhaoLi74/SmilesPE" : name_or_path # Replace with Github repo name
            name_or_path = replace(name_or_path, "_" => "\\_")
            name = replace(tok["name"], "_" => "\\_")
            return "\\$src{$name_or_path} $name\\cite{$(join(tok["cite"], ","))}"
        end
    end

    df_out = DataFrame(
        "Tokenizer" => tok_name,
        "name_or_path" => [x["name_or_path"] for x in tokenizers],
        "Domain" => [x["domain"] for x in tokenizers],
        "Encoding" => [x["encoding"] for x in tokenizers],
        "Class" => [x["tokenizer_class"] for x in tokenizers],
        "Vocab. Size" => df_toks[:, :vocab_size],
        "Top-$k" => df_toks[:, :top_tokens],
    )

    # Format for publication
    replace!(df_out[!, "Class"], "smirk-gpe" => "GPE", "spe" => "SPE", "bpe" => "BPE", "unigram" => "Unigram", "atomwise" => "Atom-wise")
    replace!(df_out[!, "Domain"], "chemistry" => "Chemistry", "nlp" => "NLP", "nlp-science" => "NLP - Science", "nlp-matsci" => "NLP - Mat. Sci")
    replace!(df_out[!, "Encoding"], "selfies" => "SELFIES", "smiles" => "SMILES")
    df_out[!, "Top-$k"] = map(df_out[!, "Top-$k"]) do top_k_tokens
        ismissing(top_k_tokens) && return missing
        replace.(top_k_tokens, "#" => "\\#", "<unk>" => "[UNK]")
    end

    # Stack dataset specific metrics
    @info describe(df)
    rename_cols(col, x) = Symbol(col, "_", lowercase(x))
    for col in [:info_loss, :cross_entropy, :fertility]
        df_col = select(df, :dataset, :tokenizer, col)
        df_col = unstack(df_col, :dataset, col; renamecols=Base.Fix1(rename_cols, col))
        leftjoin!(df_out, df_col; on="name_or_path" => "tokenizer")
    end
    select!(df_out, Not("name_or_path"))

    fig_dir = joinpath(pkgdir(TokenizerStats), "fig")
    mkpath(fig_dir)
    open(joinpath(fig_dir, "tokenizer_summary.tex"), "w") do fid
        header = """
            \\begin{landscape}
            \\begin{table}
            \\resizebox{\\linewidth}{!}{%
            \\begin{tabular}{llllrc|cc|ccc|ccc}
                &
                &
                &
                &
                &
                &
                \\multicolumn{2}{c|}{REALSpace} &
                \\multicolumn{3}{c|}{MoleculeNet} &
                \\multicolumn{3}{c}{tmQM} \\\\
                Tokenizer &
                Domain &
                Encoding &
                Class &
                Vocab. Size &
                Top-$k &
                Fertility &
                \\(H\\) &
                Fertility &
                \\(H\\) &
                \\(D_{KL}\\) &
                Fertility &
                \\(H\\) &
                \\(D_{KL}\\) \\\\ \\hline
            """
        write(fid, header)

        for row in eachrow(df_out)
            if ismissing(row["Top-$k"])
                top_k_tokens = "---"
            else
                top_k_tokens = "\\tok{$(join(row["Top-$k"], ", "))}"
            end
            line = """
                $(row["Tokenizer"]) &
                $(row["Domain"]) &
                $(row["Encoding"]) &
                $(row["Class"]) &
                $(format("{:,d}", row["Vocab. Size"])) &
                $top_k_tokens &
                $(row["fertility_realspace"]) &
                $(row["cross_entropy_realspace"]) &
                $(row["fertility_moleculenet"]) &
                $(row["cross_entropy_moleculenet"]) &
                $(coalesce(row["info_loss_moleculenet"], "---")) &
                $(row["fertility_tmqm"]) &
                $(row["cross_entropy_tmqm"]) &
                $(coalesce(row["info_loss_tmqm"], "---")) \\\\
            """
            write(fid, line)
        end

        footer = """
            \\end{tabular}
            } % resizebox
            \\caption{\\label{tab:tokenizers}Summary statistics for the tokenizers evaluated in this paper.
                See \\cref{tab:tok_summary_cols} for an explanation of each column.
            }
        \\end{table}
        \\end{landscape}
        """
        write(fid, footer)
    end

    return df_out
end
