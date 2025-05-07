using DataFrames
using CairoMakie
using CSV: CSV
using SmirkPaperPlots
using TokenizerStats
using SmirkPaperPlots: savefig
using StatsBase
using JLD2: JLD2
using Format

stats_dir = joinpath(pkgdir(TokenizerStats), "stats")

# Tokenizer Statistics
loss_stats = SmirkPaperPlots.model_loss_stats(stats_dir)
info_loss = SmirkPaperPlots.info_loss_stats(stats_dir)
token_usage = SmirkPaperPlots.usage_stats(stats_dir)

# Tokenizer Fertility by Dataset
function fertility_summary(token_usage)
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
fertility_summary(token_usage) |> display

# Frequency of Unknown Tokens
function unk_freq(token_usage)
    tok_info = SmirkPaperPlots.tokenizers_info(stats_dir)
    token_usage = transform(token_usage,
        :dataset => ByRow(x -> x ∉ ["tmQM", "realspace"] ? "MoleculeNet" : x) => :dataset,
        :tokenizer => ByRow(x -> tok_info[x]["tokenizer_class"]) => :tokenizer_class,
    )
    combine(groupby(token_usage, [:tokenizer, :dataset])) do gdf
        unk_count = sum(gdf.unk_count)
        tokens_seen = sum(gdf.tokens_seen)
        unk_freq = unk_count / tokens_seen
        return (;
            unk_freq,
            fmt_unk_freq=format("{:.2f}%", round(100 * unk_freq; sigdigits=3)),
        )
    end
end
unk_freq(token_usage) |> display

# Normalized Entropy
function norm_entropy_summary(token_usage)
    token_usage = subset(token_usage,
        :dataset => ByRow(==("realspace")),
    )
    # Combine splits
    df = combine(groupby(token_usage, :tokenizer)) do gdf
        return (; normalized_entropy=mean(gdf.normalized_entropy, fweights(gdf.tokens_seen)))
    end
    tok_info = SmirkPaperPlots.tokenizers_info(stats_dir)
    df = transform(df,
        :tokenizer => ByRow(x -> tok_info[x]["tokenizer_class"]) => :tokenizer_class,
        :tokenizer => ByRow(x -> tok_info[x]["domain"]) => :domain,
    )
    # subset!(df, :domain => ByRow(==("chemistry")))
    df.tokenizer_class .= replace.(df.tokenizer_class, "smirk-gpe" => "smirk")
    # Combine over tokenizer classes
    df = combine(groupby(df, [:tokenizer_class, :domain])) do gdf
        return (;
            norm_entropy_avg=mean(gdf.normalized_entropy),
            norm_entropy_std=std(gdf.normalized_entropy),
            norm_entropy_n=nrow(gdf),
        )
    end
    sort!(df, [:norm_entropy_avg, :norm_entropy_std, :norm_entropy_n])
    return df
end
norm_entropy_summary(token_usage) |> display


# Transformer Models
dfp, dff, dft = SmirkPaperPlots.transformer_models(stats_dir)

# Fixed Effects models for results
fe_models = SmirkPaperPlots.ngram_vs_transformer_fits(stats_dir, loss_stats, dfp, dff, dft)
JLD2.jldsave("fe_models.jld2"; fe_models)


# Tokenizer Summary
df_tok = SmirkPaperPlots.tokenizer_summary(stats_dir; k=5)

# Tokenizer Summary
SmirkPaperPlots.report_tokenizer_summary_stats(stats_dir, loss_stats, info_loss, token_usage; k=3)

with_theme(SmirkPaperPlots.theme()) do
    # Token Usage
    savefig("token_usage", SmirkPaperPlots.figure_token_usage(stats_dir))
    savefig("oov_rate", SmirkPaperPlots.figure_oov_rate(stats_dir))
    savefig("jaccard", SmirkPaperPlots.figure_jaccard(stats_dir))

    # Transformers vs. N-Grams
    df = SmirkPaperPlots.ngram_vs_transformer_fits(stats_dir, loss_stats, dfp)
    savefig("ngram_vs_transformer", SmirkPaperPlots.figure_ngram_vs_transformer(stats_dir, df))

    # Transformer Model Summary
    f, df = SmirkPaperPlots.figure_tf_finetune(stats_dir, dff, dft)
    savefig("tf_finetune", f)
    CSV.write(joinpath("stats/tf_model_summary.csv"), df)

    # N-Gram Analysis
    savefig("ngram_fits", SmirkPaperPlots.figure_ngram_fits(loss_stats, stats_dir))
    savefig("kl_v_info_loss", SmirkPaperPlots.figure_kl_v_info_loss(stats_dir, loss_stats, info_loss, df))

    # Example n-gram predictions
    compunds = [
        "caffeine" => "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "lsd" => "CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4c3c(ccc4)C2=C1)C",
        "cortisol" => "O=C4\\C=C2/[C@]([C@H]1[C@@H](O)C[C@@]3([C@@](O)(C(=O)CO)CC[C@H]3[C@@H]1CC2)C)(C)CC4",
    ]
    for (name, smi) in compunds
        for direction in [:forward, :bidirectional]
            savefig(
                "log_prob_$(direction)_$name",
                SmirkPaperPlots.figure_ngram_prediction(smi, stats_dir; direction)
            )
        end
    end
end

