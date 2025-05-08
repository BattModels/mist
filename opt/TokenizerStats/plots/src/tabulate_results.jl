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
        metric = dataset in ["esol", "freesolv", "lipo", "rmse"] ? "rmse" : "mae"
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
        if metric["metric"] == name && metric["split"] == split && coalesce(metric["tok_group"] == tok_group, false) && coalesce(metric["type"] == type, false)
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

function df_ngrams_vs_transformer(stats_dir, loss_stats, dfp, dff, dft)
    tokenizers = tokenizers_info(stats_dir)
    df_ng = subset(loss_stats,
        :split => ByRow(==("val")),
        :ngram => ByRow(==(5)),
        :tokenizer => ByRow(x -> haskey(tokenizers, x)),
    )
    transform!(df_ng,
        :tokenizer => ByRow(x -> tokenizers[x]["tokenizer_class"]) => :tokenizer_class,
        :tokenizer => ByRow(x -> tokenizers[x]["encoding"]) => :encoding,
    )
    select!(df_ng, :tokenizer, :tokenizer_class, :dataset, :encoding, :finetuned, :samples, :loss_per_token_moments)

    # Combine Molecular Foundation Models
    dfp = select(dfp, :id => :pretrained_id, :id, :tokenizer, :encoding, :tokenizer_class, :val_loss, :train_loss)
    dfp.task .= "mlm"
    dfp.metric .= "cross-entropy"
    dff = subset(dff,
        :frozen => ByRow(!),
        [:encoding, :pretrained_encoding] => ByRow(==),
        :dataset => ByRow(!=("muv")),
    )
    select!(dff, Not([:frozen, :train_oov_loss, :val_oov_loss, :train_loss, :val_loss, :step]))
    select!(dff, Not([:pretrained_encoding]))

    # Get preferred MoleculeNet Metrics
    dft = subset(dft, :tok_group => ByRow(==("all")), :channel => ByRow(isnothing))
    select!(dft, :ckpt_id, :metric, :mean, :std)
    leftjoin!(dff, dft; on=[:id => :ckpt_id, :metric])
    dropmissing!(dff)


    return df_ng, dfp, dff
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

nobs_nonwts(model) = size(model.model.pp.X, 1)

zscore(x) = (x .- mean(x)) ./ std(x)

function ngram_vs_transformer_fits(stats_dir, loss_stats, dfp, dff, dft)
    df_ng, df_p, df_f = df_ngrams_vs_transformer(stats_dir, loss_stats, dfp, dff, dft)
    contrasts = Dict(
        :tokenizer_class => EffectsCoding(; base="atomwise"),
        :encoding => EffectsCoding(; base="smiles"),
    )

    # Pretraining models
    models = []
    df_ng.val_loss = mean.(df_ng.loss_per_token_moments)
    df_ng.ng_loss_std = std.(df_ng.loss_per_token_moments)
    df_ng.wts = inv.(var.(df_ng.loss_per_token_moments))
    df_ng_pt = subset(df_ng, :dataset => ByRow(==("realspace")))
    model = lm(@formula(val_loss ~ 1 + tokenizer_class + encoding), df_ng_pt;
        contrasts,
        wts=aweights(df_ng_pt.wts),
    )
    df_ng_pt.ng_est_loss = predict(model)

    # Dataframe for predictions
    df_predict = leftjoin(
        select(df_ng_pt, :tokenizer, :val_loss => :ng_loss_avg, :ng_loss_std, :encoding),
        select(df_p, :tokenizer, :val_loss => :fm_loss_avg, :encoding);
        on=[:tokenizer, :encoding],
    )
    df_predict.fm_loss_std .= missing
    df_predict.finetuned .= false
    df_predict.dataset .= "realspace"

    push!(models, (; model, dataset="realspace", ngram=true, metric="CE"))
    model = lm(@formula(val_loss ~ 1 + tokenizer_class + encoding), df_p; contrasts)
    push!(models, (; model, dataset="realspace", ngram=false, metric="CE"))

    for dataset in unique(df_f.dataset)
        df_f_fm = subset(df_f, :dataset => ByRow(==(dataset)))
        task = first(df_f_fm.task)

        # Foundation Model
        df_f_fm.val_loss = df_f_fm.mean
        df_f_fm.wts = inv.(df_f_fm.std .^ 2)
        model = lm(
            @formula(val_loss ~ 1 + tokenizer_class + encoding), df_f_fm;
            contrasts,
            wts=aweights(df_f_fm.wts),
        )
        push!(models, (; model, dataset, ngram=false, metric=first(df_f_fm.metric)))

        # NGram
        for finetuned in [true, false]
            df_f_ng = subset(df_ng, :dataset => ByRow(==(dataset)), :finetuned => ByRow(==(finetuned)))
            nrow(df_f_ng) == 0 && continue
            df_f_ng.val_loss = mean.(df_f_ng.loss_per_token_moments)
            df_f_ng.wts = inv.(var.(df_f_ng.loss_per_token_moments))
            model = lm(
                @formula(val_loss ~ 1 + tokenizer_class + encoding), df_f_ng;
                contrasts,
                wts=aweights(df_f_ng.wts),
            )
            push!(models, (; model, dataset, ngram=true, finetuned, metric="CE"))

            ds_predict = leftjoin(
                select(df_f_ng, :tokenizer, :val_loss => :ng_loss_avg, :ng_loss_std, :encoding),
                select(df_f_fm, :tokenizer, :val_loss => :fm_loss_avg, :std => :fm_loss_std, :encoding);
                on=[:tokenizer, :encoding],
            )
            ds_predict.dataset .= dataset
            ds_predict.finetuned .= finetuned
            df_predict = vcat(df_predict, ds_predict)
        end

    end

    models = map(models) do m
        m = haskey(m, :finetuned) ? m : (; m..., finetuned=false)
    end

    df_model = DataFrame(models)
    transform!(df_model,
        :model => ByRow(m -> cor(response(m), predict(m))) => :r2,
        :model => ByRow(m -> corspearman(response(m), predict(m))) => :spearman,
        :model => ByRow(nobs_nonwts) => :nobs,
    )
    return df_model, df_predict
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
