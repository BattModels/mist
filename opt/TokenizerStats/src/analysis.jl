function ngram_perf()
    rows = []
    stats_dir = joinpath(@__DIR__, "..", "stats")
    for file in find(stats_dir, r".*\.bson")
        data = BSON.load(file)
        file = relpath(file, stats_dir)
        tokenizer = joinpath(splitpath(file)[1:end-1])
        dataset = first(splitext(basename(file)))
        haskey(data, :ngram_log_odds) || continue
        length(data[:ngram_log_odds]) == 5 || continue
        for (ngram, log_odds) in enumerate(data[:ngram_log_odds])
            push!(rows, (;
                tokenizer,
                dataset,
                vocab_size = data[:tokenizer][:vocab_size],
                samples = data[:samples],
                out_of_vocab = data[:out_of_vocab],
                ngram,
                log_odds,
            ))
        end
    end
    return DataFrame(rows)
end

function info_loss_stats()
    rows = []
    stats_dir = joinpath(@__DIR__, "..", "stats")
    for file in find(stats_dir, r".*info_loss\.bson")
        data = BSON.load(file)
        file = relpath(file, stats_dir)
        tokenizer = joinpath(splitpath(file)[1:end-1])
        dataset = first(split(basename(file), "_"; limit=2))
        for (ngram, loss) in enumerate(data[:info_loss])
            n_zero = loss[:extrema][:nmin]
            n_nonzero = data[:samples] - n_zero
            push!(rows, (;
                tokenizer,
                dataset,
                ngram,
                split=:val,
                vocab_size = data[:tokenizer][:vocab_size],
                samples = data[:samples],
                avg_info_loss = first(loss[:moments]),
                max_info_loss = loss[:extrema][:max],
                n_nonzero,
                nonzero_avg_info_loss = (data[:samples]*first(loss[:moments])) / n_nonzero,
            ))
        end
    end
    return DataFrame(rows)
end

function model_loss_stats()
    rows = []
    stats_dir = joinpath(@__DIR__, "..", "stats")
    for file in find(stats_dir, r".*model_loss\.bson")
        data = BSON.load(file)
        file = relpath(file, stats_dir)
        tokenizer = joinpath(splitpath(file)[1:end-1])
        dataset = first(split(basename(file), "_"; limit=2))
        for split in [:train, :val, :test]
            split ∉ keys(data) && continue
            for (ngram, loss) in enumerate(data[split])
                push!(rows, (;
                    tokenizer,
                    dataset,
                    split,
                    ngram,
                    vocab_size = data[:tokenizer][:vocab_size],
                    samples = 1,
                    avg_model_loss = first(loss[:moments]),
                    max_model_loss = loss[:extrema][:min],
                ))
            end
        end
    end
    return DataFrame(rows)
end

function avg_molnet_info_loss()
    df = TokenizerStats.info_loss_stats()
    return combine(groupby(df, [:tokenizer, :split, :ngram])) do gdf
        # Information Loss Statistics
        avg_info_loss = mean(gdf.avg_info_loss, Weights(gdf.samples))
        nonzero = subset(gdf, :n_nonzero => ByRow(x -> x > 0))
        avg_nonzero_info_loss = mean(nonzero.avg_info_loss, Weights(nonzero.n_nonzero))
        n_nonzero=sum(nonzero.n_nonzero)
        samples=sum(gdf.samples)
        return (;
            vocab_size=first(gdf.vocab_size),
            avg_info_loss,
            avg_perplexity=exp(avg_info_loss),
            samples,
            avg_nonzero_info_loss,
            nonzero_perplexity=exp(avg_nonzero_info_loss),
            n_nonzero,
            nonzero_rate = n_nonzero / samples,
        )
    end
end

function find_oov_samples(results::Dict)
    oov_samples = Dict{String, Set{String}}()
    for (name, tok_stats) in results
        oov_samples[name] = Set{String}()
        for (k, v) in tok_stats
            if startswith(k, "MoleculeNet")
                foreach(Base.Fix1(push!, oov_samples[name]), v["oov_samples"])
            end
        end
    end
    examples = Dict{String, Set{String}}()
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
