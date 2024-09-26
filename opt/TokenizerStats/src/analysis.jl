function classify_tokenizer(name::String)
    if startswith(name, "smirk")
        return :ours
    elseif name in ["Xenova/gpt-4o", "google/gemma-7b"] || startswith(name, "meta-llama")
        return :nlp
    elseif startswith(name, "sagawa") || name in ["ChangwenXu98/TransPolymer"]
        return :nlp_based
    else
        return :atomic
    end
end

function usage_stats()
    rows = []
    stats_dir = joinpath(@__DIR__, "..", "stats")
    for file in union(find(stats_dir, r"realspace_v4_dev2?\.bson"), find(stats_dir, r"[a-z0-9]+\.bson"))
        data = BSON.load(file)
        file = relpath(file, stats_dir)
        tokenizer = joinpath(splitpath(file)[1:end-1])
        dataset = first(splitext(basename(file)))
        for split in [:train, :val, :test]
            split ∉ keys(data) && continue
            Set(keys(data[split])) >= Set([:samples, :out_of_vocab, :fertility]) || continue
            fertility = mean_std_countmap(data[split][:fertility])
            nunique = mean_std_countmap(data[split][:fertility])
            push!(rows, (;
                file,
                tokenizer,
                dataset,
                split,
                samples = data[split][:samples],
                out_of_vocab = data[split][:out_of_vocab],
                avg_fertility = first(fertility),
                std_fertility = last(fertility),
                max_fertility = maximum(keys(data[split][:fertility])),
                avg_nunique = first(nunique),
                std_nunique = last(nunique),
            ))
        end
    end
    df = DataFrame(rows)
    replace!(df.dataset, "realspace_v4_dev2" => "realspace_v4_dev")
    return df
end

function mean_std_countmap(x::AbstractDict)
    v = keys(x)
    c = values(x)
    return StatsBase.mean_and_std(collect(v), Weights(collect(c)))
end

function info_loss_stats()
    rows = []
    stats_dir = joinpath(@__DIR__, "..", "stats")
    for file in find(stats_dir, r".*_info_loss\.bson")
        data = BSON.load(file)
        file = relpath(file, stats_dir)
        tokenizer = joinpath(splitpath(file)[1:end-2])
        dataset = basename(dirname(file))
        for (ngram, loss) in enumerate(data[:info_loss])
            n_zero = loss[:extrema][:nmin]
            n_nonzero = data[:samples] - n_zero
            push!(rows, (;
                tokenizer,
                ref_tokenizer = data[:ref_tokenizer][:name],
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
    df = DataFrame(rows)
    replace!(df.dataset, "realspace_v4_dev2" => "realspace_v4_dev")
    return df
end

function model_loss_stats()
    rows = []
    stats_dir = joinpath(@__DIR__, "..", "stats")
    for file in find(stats_dir, r".*model_loss\.bson")
        data = BSON.load(file)
        tokenizer = data[:ref_tokenizer][:name]
        dataset = basename(dirname(file))
        training_data = match(r"(.*?)_model_loss\.bson", basename(file)).captures[1]
        for split in [:train, :val, :test]
            split ∉ keys(data) && continue
            split_data = data[split]
            for ngram in 1:5
                push!(rows, (;
                    tokenizer,
                    training_dataset = training_data,
                    dataset,
                    split,
                    ngram,
                    vocab_size = data[:tokenizer][:vocab_size],
                    samples = split_data[:samples],
                    avg_model_loss = first(split_data[:kld][ngram][:moments]),
                    stderr_model_loss = sqrt(split_data[:kld][ngram][:moments][2]) / sqrt(split_data[:samples]),
                ))
            end
        end
    end
    df = DataFrame(rows)
    replace!(df.dataset, "realspace_v4_dev2" => "realspace_v4_dev")
    replace!(df.training_dataset, "realspace_v4_dev2" => "realspace_v4_dev")
    return df
end

function avg_molnet_info_loss()
    df = TokenizerStats.info_loss_stats()
    subset!(df, :dataset => ByRow(!=("realspace_v4_dev")))
    return combine(groupby(df, [:tokenizer, :ref_tokenizer, :split, :ngram])) do gdf
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

function oov_stats()
    data = JSON.parsefile(joinpath(@__DIR__, "..", "stats-atomic.json"))
    rows = []
    for (tokenizer, tok_data) in pairs(data)
        for (dataset, stats) in pairs(tok_data)
            ds_name = startswith(dataset, "MoleculeNet") ? lowercase(split(dataset,"/")[end]) : dataset
            ds_name = replace(ds_name, "lipophilicity" => "lipo")
            push!(rows, (;
                tokenizer,
                molnet = startswith(dataset, "MoleculeNet"),
                dataset = ds_name,
                samples = stats["nobs"],
                n_oov = stats["oov"],
                oov_samples = stats["oov_samples"],
            ))
        end
    end
    return DataFrame(rows)
end

function tokenizer_stats()
    tokenizers = keys(JSON.parsefile(joinpath(@__DIR__, "..", "stats-atomic.json"))) |> collect
    smirk = load_tokenizer("smirk")
    filter!(x -> !occursin("smirk-gpe", x), tokenizers)
    rows = []
    for tokenizer in tokenizers
        @info tokenizer
        tok = load_tokenizer(tokenizer)
        class = classify_tokenizer(tokenizer)

        # Count carbon containing tokens
        local n_carbon_tokens
        local carbon_tokens
        if class != :nlp
            vocab = pyconvert(Vector{String}, values(tok.get_vocab()))
            carbon_tokens = filter(x -> has_element(smirk, x), vocab)
            n_carbon_tokens = length(carbon_tokens)
        else
            carbon_tokens = missing
            n_carbon_tokens = missing
        end
        push!(rows, (;
            tokenizer,
            vocab_size = pyconvert(Int, length(tok)),
            carbon_tokens,
            n_carbon_tokens,
        ))
    end
    df = DataFrame(rows)
    df = leftjoin(df, select(top_k_tokens(; k=5), [:tokenizer, :top_k_tokens]), on=:tokenizer)
    return df
end

function tokenizer_jaccard(tokenizers::Vector{String})
    tok_tokens = map(tokenizers) do tokenizer
        name_or_path = startswith(tokenizer, "smirk-gpe") ? "./" * tokenizer : tokenizer
        tok = load_tokenizer(name_or_path)
        vocab_size = pyconvert(Int, length(tok))
        pytokens = map(id -> tok.decode([id]), range(0; length=vocab_size))
        tokens = pyconvert(Vector{String}, pytokens)
        return tokenizer => tokens
    end |> Dict
    return tokenizer_jaccard(tok_tokens)
end

function tokenizer_jaccard(tok_vocab::Dict{String})
    k = collect(keys(tok_vocab))
    J = Matrix{Float64}(undef, length(k), length(k))
    for i in 1:length(k)
        J[i, i] = jaccard(tok_vocab[k[i]], tok_vocab[k[i]])
        @assert J[i, i] == 1
        for j in i+1:length(k)
            J[i, j] = jaccard(tok_vocab[k[i]], tok_vocab[k[j]])
            J[j, i] = J[i, j]
        end
    end
    return J, k
end

jaccard(a::AbstractVector, b::AbstractVector) = jaccard(Set(a), Set(b))
jaccard(a::AbstractSet, b::AbstractSet) = length(intersect(a, b)) / length(union(a, b))

function has_element(smirk::Py, smiles::String; element::String="C")
    smiles = replace(smiles, "▁" => "")
    local tokens
    try
        tokens = pyconvert(Vector{String}, smirk.tokenize(smiles))
    catch
        # Smirk's support for unicode is limited, ignore errors
        @warn "failed to tokenize" smiles
        return false
    end
    any(x -> occursin(r"\[[A-Z]*?]", x), tokens) && return false # Don't count tokens smirk can't parse
    if length(tokens) == 1 && tokens[1] == element
        return true
    elseif !occursin(r"^\[.*?]$", smiles)
        return false # Non-bracketed token -> skip
    end
    n = 0
    for i in 1:length(tokens)
        gram = tokens[i:min(i+1, length(tokens))]
        if length(gram) == 1 && gram[1] == element
            n += 1
        elseif length(gram) == 2
            if gram[1] == element && !islowercase(gram[2][1])
                n += 1
            end
        end
    end
    return n == 1 # Don't count merged tokens
end

function top_k_tokens(; k=5)
    statsdir = joinpath(@__DIR__, "..", "stats")
    results = find(statsdir, r"realspace_v4_dev2?\.bson")
    rows = []
    for file in results
        tokenizer = joinpath(splitpath(relpath(file, statsdir))[1:end-1])
        data = BSON.load(file)

        # Find the top k tokens
        usage = data[:train][:ngrams][1]
        usage = Dict{Int, Int}(only(k) => v for (k, v) in pairs(usage))
        vocab_size = data[:tokenizer][:vocab_size]
        c_token = collate_token_usage(usage, vocab_size; smoothing=0)
        p_token = c_token ./ sum(c_token)
        ids = sortperm(p_token; rev=true) .- 1 # token ids are 0-indexed
        top_k = ids[1:k]

        # Get the tokens
        tokenizer = startswith(tokenizer, "smirk-gpe") ? "./" * tokenizer : tokenizer
        tok = load_tokenizer(tokenizer)
        top_k_tokens = map( id -> pyconvert(String, tok.decode([id])), top_k)
        push!(rows, (; tokenizer, top_k, top_k_tokens))
    end
    return DataFrame(rows)
end
