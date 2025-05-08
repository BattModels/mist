function usage_stats(stats_dir)
    rows = []
    for file in find(stats_dir, r"usage.jld2$")
        jldopen(file) do data
            file = relpath(file, stats_dir)
            tokenizer = joinpath(splitpath(file)[1:end-2])
            dataset = splitpath(file)[end-1]
            dataset = dataset == "tmqm" ? "tmQM" : dataset
            unk_id = data["tokenizer"][:unk_token_id]
            vocab_size = data["tokenizer"][:vocab_size]

            for split in ["train", "val", "test"]
                split ∉ keys(data) && continue
                Set(keys(data[split])) >= Set(["samples", "out_of_vocab", "fertility"]) || continue
                samples = data[split]["samples"]
                fertility = CountMap(data[split]["fertility"], samples)
                unigram = data[split]["ngrams"]["1"]
                unk_count = get(unigram, (unk_id,), 0)
                tokens_seen = sum(values(unigram))
                nunique = CountMap(data[split]["fertility"], samples)

                token_prob = values(unigram) ./ tokens_seen
                entropy = sum(token_prob) do p
                    -p * log(p)
                end
                normalized_entropy = entropy / log(data["tokenizer"][:vocab_size])

                divergence = 0.5 * sum(token_prob) do p
                    abs(p - (1 / vocab_size))
                end

                c_all = collate_token_usage(unigram, vocab_size; smoothing=0)
                f95_all = tokens_seen .* percentile(c_all ./ tokens_seen, 5)
                p_smooth = @. (c_all + 1) / (tokens_seen + length(c_all))
                f95_smooth = tokens_seen .* percentile(p_smooth, 5)
                f95 = tokens_seen .* percentile(token_prob, 5)

                push!(rows, (;
                    file,
                    tokenizer,
                    dataset,
                    split,
                    samples,
                    out_of_vocab=data[split]["out_of_vocab"],
                    fertility,
                    entropy,
                    divergence,
                    f95,
                    f95_all,
                    f95_smooth,
                    normalized_entropy,
                    nunique,
                    unk_count,
                    tokens_seen,
                ))
            end
        end
    end
    df = DataFrame(rows)
    return df
end

StatsBase.mean(s::OnlineStats.CountMap) = StatsBase.mean(collect(keys(s)), Weights(collect(values(s))))
StatsBase.std(s::OnlineStats.CountMap) = StatsBase.std(collect(keys(s)), Weights(collect(values(s))))
StatsBase.quantile(s::OnlineStats.CountMap, p) = StatsBase.quantile(collect(keys(s)), Weights(collect(values(s))), p)

function mean_std_countmap(x::AbstractDict)
    v = keys(x)
    c = values(x)
    return StatsBase.mean_and_std(collect(v), Weights(collect(c)))
end

function info_loss_stats(stats_dir)
    rows = []
    for file in find(stats_dir, r"character_info_loss\.jld2$")
        try
            jldopen(file) do data
                file = relpath(file, stats_dir)
                tokenizer = joinpath(splitpath(file)[1:end-2])
                dataset = basename(dirname(file))
                dataset = dataset == "tmqm" ? "tmQM" : dataset
                for (ngram, loss) in enumerate(data["info_loss"])
                    n_zero = loss[:extrema][:nmin]
                    n_nonzero = data["samples"] - n_zero
                    loss_moments = OnlineStats.Moments(loss[:moments], EqualWeight(), data["samples"])
                    push!(rows, (;
                        tokenizer,
                        ref_tokenizer=data["ref_tokenizer"][:name],
                        dataset,
                        ngram,
                        vocab_size=data["tokenizer"][:vocab_size],
                        samples=data["samples"],
                        info_loss_moments=loss_moments,
                        avg_info_loss=mean(loss_moments),
                        std_info_loss=std(loss_moments),
                        max_info_loss=loss[:extrema][:max],
                        n_nonzero,
                        nonzero_avg_info_loss=(data["samples"] * mean(loss_moments)) / n_nonzero,
                    ))
                end
            end
        catch err
            @info "failed to load" file err
        end
    end
    df = DataFrame(rows)
    return df
end

function model_loss_stats(stats_dir)
    rows = []
    for file in find(stats_dir, r"/model_loss.*\.jld2$")
        jldopen(file) do data
            tokenizer = data["ref_tokenizer"][:name]
            tokenizer = dirname(tokenizer) == "." ? basename(tokenizer) : tokenizer
            dataset = basename(dirname(file))
            finetuned = occursin(dataset, basename(file))
            dataset = lowercase(dataset) == "tmqm" ? "tmQM" : dataset
            for split in ["train", "val", "test"]
                split ∉ keys(data) && continue
                split_data = data[split]
                for ngram in 1:5
                    cross_entropy = OnlineStats.Moments(split_data[:cross_entropy][ngram][:moments], EqualWeight(), split_data[:samples])
                    cross_entropy_per_token = OnlineStats.Moments(split_data[:cross_entropy_per_token][ngram][:moments], EqualWeight(), split_data[:samples])

                    push!(rows, (;
                        tokenizer,
                        dataset,
                        finetuned,
                        split,
                        ngram,
                        vocab_size=data["tokenizer"][:vocab_size],
                        samples=split_data[:samples],
                        loss_moments=cross_entropy,
                        loss_per_token_moments=cross_entropy_per_token,
                        avg_model_loss=mean(cross_entropy),
                        std_model_loss=std(cross_entropy),
                        avg_model_token_loss=mean(cross_entropy_per_token),
                        std_model_token_loss=std(cross_entropy_per_token),
                    ))
                end
            end
        end
    end
    df = DataFrame(rows)
    return df
end


function find_oov_samples(results::Dict)
    oov_samples = Dict{String,Set{String}}()
    for (name, tok_stats) in results
        oov_samples[name] = Set{String}()
        for (k, v) in tok_stats
            if startswith(k, "MoleculeNet")
                foreach(Base.Fix1(push!, oov_samples[name]), v["oov_samples"])
            end
        end
    end
    examples = Dict{String,Set{String}}()
    for (name, samples) in pairs(oov_samples)
        for sample in samples
            examples[sample] = union(get(examples, sample, Set{String}()), [name])
        end
    end
    df = DataFrame(smi=collect(keys(examples)), tokenizer=collect(values(examples)))
    transform!(df, :tokenizer => ByRow(length) => :ntokenizers)
    sort!(df, :ntokenizers; rev=true)
    return df
end

function fertility_summary(df_tokenizer, df_usage)
    df_usage = transform(df_usage,
        :dataset => ByRow(x -> lowercase(x) in ["tmqm", "realspace"] ? x : "MoleculeNet") => :dataset
    )
    df_usage = combine(groupby(df_usage, [:tokenizer, :dataset, :encoding])) do gdf
        return (;
            loss_per_token_moments=reduce(merge, gdf.loss_per_token_moments),
            loss_moments=reduce(merge, gdf.loss_per_token_moments),
        )
    end
    return df_usage
end
