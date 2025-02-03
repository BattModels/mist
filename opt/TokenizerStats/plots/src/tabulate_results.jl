"""
    dfp, dff, dft = transformer_models(stats_dir, cache; sweep_file)

Collate wandb results into pretraining/finetune/test dataframes for further analysis.
By default load all results (can be trickier to filter), but can be restricted to only the
relevant runs (i.e. those in a sweep)
"""
function transformer_models(
    stats_dir;
    cache=joinpath(pkgdir(TokenizerStats), "models", "wandb-export"),
    sweep_file=nothing,
)
    # Get the pretrained models
    cache = abspath(cache)
    runs = []
    pretrained = map(readdir(joinpath(cache, "pretraining"); join=true)) do file
        data = JSON.parsefile(file; null=missing)
        tokenizer = data["data"]["tokenizer"]
        if !ismissing(tokenizer) && startswith(tokenizer, "/")
            tokenizer = basename(tokenizer)
        end
        return (;
            id=data["id"],
            state=data["state"],
            tokenizer,
            model_class=data["model"]["class_path"],
            encoding=data["data"]["encoding"],
            d_model=data["model"]["d_model"],
            d_ff=data["model"]["d_ff"],
            n_layers=data["model"]["n_layers"],
            n_heads=data["model"]["n_heads"],
            model_size=data["model"]["model_size"],
            val_loss=data["metrics"]["val_loss_best"],
            train_loss=data["metrics"]["train_loss_best"],
            molecules_seen=data["trainer"]["effective_batch_size"] * data["trainer"]["num_training_steps"],
            tokens_seen=data["trainer"]["tokens"],
        )
    end |> DataFrame
    subset!(pretrained, :state => ByRow(==("finished")))
    dropmissing!(pretrained)
    pretrained.d_model = Int.(pretrained.d_model)

    # Add tokenizer info
    tokenizers = tokenizers_info(stats_dir)
    pretrained.tokenizer_class = map(name_or_path -> tokenizers[name_or_path]["tokenizer_class"], pretrained.tokenizer)
    dropmissing!(pretrained)

    # Get the finetuned models
    fintuned = []
    for file in readdir(joinpath(cache, "finetuning"); join=true)
        data = JSON.parsefile(file; null=missing)
        pretrained_id = data["model"]["encoder_id"]
        task = get(data["model"], "task", missing)
        !ismissing(task) || continue
        data["state"] == "finished" || continue
        push!(fintuned, (;
            id=data["id"],
            pretrained_id,
            task,
            dataset=data["data"]["dataset"],
            encoding=data["data"]["encoding"],
            step=data["trainer"]["step"],
            frozen=data["model"]["freeze_encoder"],
            task_metrics(task, data["metrics"]; dataset=data["data"]["dataset"])...
        ))
    end
    finetuned = DataFrame(fintuned)
    dropmissing!(finetuned, :pretrained_id)

    # Add pretraining info
    leftjoin!(finetuned,
        select(pretrained,
            :id => :pretrained_id,
            :tokenizer,
            :tokenizer_class,
            :encoding => :pretrained_encoding,
            :val_loss => :pretrained_val_loss
        ),
        on=:pretrained_id
    )

    # Test Results
    runs = []
    for file in readdir(joinpath(cache, "test"); join=true)
        data = JSON.parsefile(file)
        for metric in data["metrics"]
            metric["value"] = "Infinity" == metric["value"] ? Inf : metric["value"]
            push!(runs, (;
                id=data["id"],
                ckpt_id=data["model"]["ckpt_id"],
                NamedTuple((Symbol(k) => v for (k, v) in pairs(metric)))...,
            ))
        end
    end
    test = DataFrame(runs)

    # Add Test Results to finetuning data
    subset!(test,
        :type => ByRow(==("last")),
        :split => ByRow(==("test")),
        :tok_group => ByRow(!isnothing),
        :metric => ByRow(x -> x ∉ ["global_step", "loss_step"]),
    )
    select!(test, Not([:split, :type]))
    transform!(test, :bootstrap => ByRow(x -> something(x, "mean")) => :bootstrap)
    test = unstack(test, [:id, :ckpt_id, :metric, :tok_group, :channel], :bootstrap, :value)
    test.tok_group = coalesce.(test.tok_group)
    dropmissing!(test, :mean)

    # Restrict to relevant runs
    if sweep_file !== nothing
        return filter_runs(pretrained, finetuned, test; sweep_file)
    end

    return pretrained, finetuned, test
end

function copy_sweep_results(dfp::DataFrame, dff::DataFrame, dft::DataFrame,
    src_cache=joinpath(pkgdir(TokenizerStats), "..", "..", ".cache", "wandb-export"),
    dst_cache=joinpath(pkgdir(TokenizerStats), "models", "wandb-export"),
)
    # Copy to data drop
    copy_sweep_results("pretraining", dfp.id, src_cache, dst_cache)
    copy_sweep_results("finetuning", dff.id, src_cache, dst_cache)
    copy_sweep_results("test", dft.id, src_cache, dst_cache)

end

function copy_sweep_results(type, ids, src, dst=joinpath(pkgdir(TokenizerStats), "models", "wandb-export"))
    mkpath(joinpath(dst, type))
    for id in ids
        cp(joinpath(src, type, id * ".json"), joinpath(joinpath(dst, type, id * ".json")); force=true)
    end
    return nothing
end

function task_metrics(task, metrics; dataset=nothing)
    if task == "regression"
        metric = "r2"
    elseif task == "binary"
        metric = dataset == "muv" ? "avg-precision" : "auroc"
    else
        metric = missing
    end
    return (;
        metric,
        train_loss=get_metric(metrics, metric; split="train", tok_group="all"),
        train_oov_loss=get_metric(metrics, metric; split="train", tok_group="oov"),
        val_loss=get_metric(metrics, metric; split="val", tok_group="all"),
        val_oov_loss=get_metric(metrics, metric; split="val", tok_group="oov"),
    )
end

function get_metric(metrics::Vector, name::String; split, tok_group="all", type="best")
    for metric in metrics
        if metric["metric"] == name && metric["split"] == split && metric["tok_group"] == tok_group && metric["type"] == type
            return metric["value"]
        end
    end
    return missing
end

function filter_runs(dfp, dff, dft; sweep_file)
    data = JSON.parsefile(sweep_file)
    dfp = subset(dfp, :id => ByRow(x -> x ∈ keys(data)))
    ids_finetune = []
    ids_test = []
    for (_, ds_runs) in pairs(data)
        for (_, runs) in pairs(ds_runs)
            push!(ids_finetune, runs["finetune"])
            push!(ids_test, runs["test"])
        end
    end

    dff = subset(dff, :id => ByRow(x -> x ∈ ids_finetune))
    dft = subset(dft, :id => ByRow(x -> x ∈ ids_test))

    return dfp, dff, dft
end

"""
Identify finetuning runs for a set of pretraining runs
"""
function finetuning_sweeps(df)
    df = deepcopy(df)
    datasets = levels(df.dataset)
    df = unstack(df[!, [:id, :pretrained_id, :dataset]], :dataset, :id; combine=first)
    runs = Dict{String,Dict{String,Union{Missing,String}}}()
    for row in eachrow(df)
        items = Dict(pairs(row))
        pop!(items, :pretrained_id)
        runs[row.pretrained_id] = Dict(string(k) => get(items, Symbol(k), missing) for k in datasets)
    end
    return runs
end

""" Link Pretrained -> finetune -> test runs """
function link_training_runs(runs)
    dfp, dff, dft = transformer_models()
    subset!(dff, :frozen => ByRow(!))
    linked = Dict()
    for (id, fine) in pairs(runs)
        linked[id] = Dict()
        for (ds, id_fine) in pairs(fine)
            ds == "tmQM-tm-split" && continue
            test_runs = unique(subset(dft, :ckpt_id => ByRow(==(id_fine))).id)
            if length(test_runs) > 1
                @warn "duplicate tests for $id_fine - $ds" test_runs
            end
            id_test = isempty(test_runs) ? missing : first(test_runs)
            linked[id][ds] = Dict("finetune" => id_fine, "test" => id_test)
        end
    end
    return linked
end

function df_ngrams_vs_transformer(stats_dir, loss_stats, dfp)
    tokenizers = tokenizers_info(stats_dir)
    val_loss = subset(loss_stats,
        :split => ByRow(==("val")),
        :dataset => ByRow(==("realspace")),
        :tokenizer => ByRow(x -> haskey(tokenizers, x)),
    )
    best_models = combine(groupby(val_loss, :tokenizer)) do gdf
        sort!(gdf, :avg_model_loss; rev=false)
        return gdf[1, :]
    end
    select!(best_models, :tokenizer, :ngram, :avg_model_loss => :ngram_loss, :avg_model_token_loss => :ngram_token_loss)

    # Merge with pretraining data
    dfp = leftjoin(dfp, best_models, on=[:tokenizer])
    sort!(dfp, :val_loss)
    dfp.tokenizer = categorical(dfp.tokenizer, levels=unique(dfp.tokenizer))
    replace!(dfp.encoding,
        "smiles" => "SMILES",
        "smiles-canonical" => "Canonical SMILES",
        "selfies" => "SELFIES"
    )
    dfp.encoding = categorical(dfp.encoding, levels=["SMILES", "Canonical SMILES", "SELFIES"])

    return dfp, best_models
end

struct LogLikelihoodRatioTest
    lr::Float64
    df::Int
end

function LogLikelihoodRatioTest(model::StatsBase.RegressionModel, null::StatsBase.RegressionModel)
    λ = -2 * (loglikelihood(null) - loglikelihood(model))
    df = dof(model) - dof(null)
    return LogLikelihoodRatioTest(λ, df)
end

HypothesisTests.pvalue(lrt::LogLikelihoodRatioTest) = pvalue(Chisq(lrt.df), lrt.lr)

function Base.show(io::IO, mime::MIME"text/plain", lrt::LogLikelihoodRatioTest)
    println(io, "λ:       $(round(lrt.lr; sigdigits=3))")
    println(io, "df:      $(lrt.df)")
    println(io, "p-value: $(round(pvalue(lrt); sigdigits=3))")
end

function ngram_vs_transformer_fits(stats_dir, loss_stats, dfp)
    df, best_models = df_ngrams_vs_transformer(stats_dir, loss_stats, dfp)

    contrasts = Dict(
        :tokenizer_class => EffectsCoding(; base="atomwise"),
        :encoding => EffectsCoding(; base="SMILES"),
    )

    # Test impact of encoding on pretraining
    null = lm(@formula(val_loss ~ 1), df)
    @show tok = lm(
        @formula(val_loss ~ 1 + tokenizer_class + encoding),
        df; contrasts
    )
    LogLikelihoodRatioTest(tok, null) |> display

    # Test impact of encoding on pretraining
    tokenizers = tokenizers_info(stats_dir)
    best_models.tokenizer_class = map(tok -> tokenizers[tok]["tokenizer_class"], best_models.tokenizer)
    best_models.encoding = map(tok -> tokenizers[tok]["encoding"], best_models.tokenizer)
    replace!(best_models.encoding,
        "smiles" => "SMILES",
        "smiles-canonical" => "Canonical SMILES",
        "selfies" => "SELFIES"
    )

    null = lm(@formula(ngram_token_loss ~ 1), best_models)
    @show tok = lm(
        @formula(ngram_token_loss ~ 1 + tokenizer_class + encoding),
        best_models; contrasts
    )
    LogLikelihoodRatioTest(tok, null) |> display


    # Predict pretraining loss using n-gram model
    null = glm(@formula(val_loss ~ 1), df, Normal(), LogLink())
    @show ngram = glm(@formula(val_loss ~ 1 + log(ngram_token_loss)), df, Normal(), LogLink(); contrasts)
    LogLikelihoodRatioTest(ngram, null) |> display
    r2(ngram.model, :devianceratio) |> display

    return df
end

function tmqm_finetune(stats_dir, dff, dft)
    df = subset(dff, :dataset => ByRow(==("tmQM")))
    select!(df, [:id, :pretrained_id, :tokenizer, :task, :dataset, :encoding])
    dft = subset(dft,
        :metric => ByRow(==("mae")),
        :tok_group => ByRow(==("all")),
    )
    select!(dft, Not([:id, :tok_group]))

    df = innerjoin(df, dft; on=:id => :ckpt_id)
    return df
end
