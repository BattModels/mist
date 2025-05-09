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
tok_info = SmirkPaperPlots.tokenizers_info(stats_dir)

function intrinsic_metrics(token_usage)
    tok_info = SmirkPaperPlots.tokenizers_info(stats_dir)
    token_usage = transform(token_usage,
        [:unk_count, :tokens_seen] => ByRow(/) => :oov_rate,
    )
    wmean = (x, s) -> mean(x, fweights(s))
    df = combine(groupby(token_usage, [:tokenizer, :dataset]),
        :fertility => (x -> mean(reduce(merge, x))) => :fertility,
        [:divergence, :samples] => wmean => :divergence,
        [:f95_all, :samples] => wmean => :f95,
        [:oov_rate, :tokens_seen] => wmean => :oov_rate,
        [:normalized_entropy, :samples] => wmean => :normalized_entropy,
        [:entropy, :samples] => wmean => :entropy,
    )
    transform!(df,
        :tokenizer => ByRow(x -> tok_info[x]["tokenizer_class"]) => :tokenizer_class,
        :tokenizer => ByRow(x -> tok_info[x]["domain"]) => :tokenizer_domain,
        :tokenizer => ByRow(x -> tok_info[x]["encoding"]) => :encoding,
    )
    return df
end
df_intrinsic = intrinsic_metrics(token_usage)
CSV.write(joinpath("stats", "intrinsic_metrics.csv"), df_intrinsic)

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
df_ng, df_p, df_f = SmirkPaperPlots.df_ngrams_vs_transformer(stats_dir, loss_stats, dfp, dff, dft)

# NGrams Stats vs. FM Performance
df_prog = SmirkPaperPlots.df_ngram_stats_v_fm_perf(tok_info, loss_stats, info_loss, df_f)
CSV.write(joinpath("stats", "ngram_stats_vs_fm.csv"), df_prog)

# Predictive and Fixed-Effect Models
fe_models, df_predict = SmirkPaperPlots.ngram_vs_transformer_fits(df_ng, df_p, df_f)
prog_models = SmirkPaperPlots.ngram_prognostic_fits(df_prog)
JLD2.jldsave(joinpath("stats", "quality_models.jld2"); fe_models, df_predict, prog_models)
CSV.write(joinpath("stats", "ngram_vs_transformer.csv"), df_predict)

# Tokenizer Summary
df_tok = SmirkPaperPlots.tokenizer_summary(stats_dir; k=5)

# Tokenizer Summary
SmirkPaperPlots.report_tokenizer_summary_stats(stats_dir, loss_stats, info_loss, token_usage; k=3)

with_theme(SmirkPaperPlots.theme()) do
    # Token Usage
    savefig("token_usage", SmirkPaperPlots.figure_token_usage(stats_dir))
    savefig("oov_rate", SmirkPaperPlots.figure_oov_rate(stats_dir))
    savefig("oov_rate_no_transcode",
        SmirkPaperPlots.figure_oov_rate(stats_dir; include_transcode_errors=false)
    )
    savefig("jaccard", SmirkPaperPlots.figure_jaccard(stats_dir))
    savefig("intrinsic_metrics", SmirkPaperPlots.figure_intrinsic(df_intrinsic))
    savefig("intrinsic_metrics_95", SmirkPaperPlots.figure_intrinsic(df_intrinsic; p=95))

    # Transformers vs. N-Grams
    savefig("prognostic", SmirkPaperPlots.figure_fe_models(fe_models, prog_models))
    savefig("prognostic_abs", SmirkPaperPlots.figure_fe_models(fe_models, prog_models; scale=:absolute))
    savefig("prognostic_std", SmirkPaperPlots.figure_fe_models(fe_models, prog_models; scale=:std))

    # Transformer Model Summary
    f, df = SmirkPaperPlots.figure_tf_finetune(stats_dir, dff, dft)
    savefig("tf_finetune", f)
    CSV.write(joinpath("stats", "tf_model_summary.csv"), df)

    # N-Gram Analysis
    savefig("ngram_fits", SmirkPaperPlots.figure_ngram_fits(loss_stats, stats_dir))
    savefig("kl_v_info_loss", SmirkPaperPlots.figure_kl_v_info_loss(tok_info, loss_stats, info_loss, df_f))

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

