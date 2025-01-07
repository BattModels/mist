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
    for file in find(stats_dir, r"usage.jld2$")
        jldopen(file) do data
            file = relpath(file, stats_dir)
            tokenizer = joinpath(splitpath(file)[1:end-2])
            dataset = splitpath(file)[end-1]
            for split in ["train", "val", "test"]
                split ∉ keys(data) && continue
                Set(keys(data[split])) >= Set([:samples, :out_of_vocab, :fertility]) || continue
                fertility = mean_std_countmap(data[split][:fertility])
                nunique = mean_std_countmap(data[split][:fertility])
                push!(rows, (;
                    file,
                    tokenizer,
                    dataset,
                    split,
                    samples=data[split][:samples],
                    out_of_vocab=data[split][:out_of_vocab],
                    avg_fertility=first(fertility),
                    std_fertility=last(fertility),
                    max_fertility=maximum(keys(data[split][:fertility])),
                    avg_nunique=first(nunique),
                    std_nunique=last(nunique),
                ))
            end
        end
    end
    df = DataFrame(rows)
    return df
end

function mean_std_countmap(x::AbstractDict)
    v = keys(x)
    c = values(x)
    return StatsBase.mean_and_std(collect(v), Weights(collect(c)))
end

function model_loss_stats()
    rows = []
    stats_dir = joinpath(@__DIR__, "..", "stats")
    for file in find(stats_dir, r"/model_loss\.jld2$")

        # Get average fertility
        usage = joinpath(dirname(file), "usage.jld2")
        if !isfile(usage)
            @warn "No usage file found for $file"
            avg_fertility = Dict{String,Float64}()
        else
            avg_fertility = jldopen(usage) do data
                usage_data = Dict{String,Float64}()
                for split in ["train", "val", "test"]
                    haskey(data, split) || continue
                    fertility = first(mean_std_countmap(data[split][:fertility]))
                    usage_data[split] = fertility
                end
                return usage_data
            end
        end

        # Get model loss
        jldopen(file) do data
            tokenizer = data["ref_tokenizer"][:name]
            dataset = basename(dirname(file))
            for split in ["train", "val", "test"]
                split ∉ keys(data) && continue
                split_data = data[split]
                for ngram in 1:5
                    local avg_model_token_loss, stderr_model_token_loss
                    try
                        avg_model_token_loss = first(split_data[:kld_per_token][ngram][:moments])
                        stderr_model_token_loss = sqrt(split_data[:kld_per_token][ngram][:moments][2]) / sqrt(split_data[:samples])
                    catch
                        @warn "falling back to loss/fertility for $file"
                        avg_model_token_loss = first(split_data[:kld][ngram][:moments]) / get(avg_fertility, split, missing)
                        stderr_model_token_loss = missing
                    end

                    push!(rows, (;
                        tokenizer,
                        dataset,
                        split,
                        ngram,
                        vocab_size=data["tokenizer"][:vocab_size],
                        samples=split_data[:samples],
                        avg_model_loss=first(split_data[:kld][ngram][:moments]),
                        stderr_model_loss=sqrt(split_data[:kld][ngram][:moments][2]) / sqrt(split_data[:samples]),
                        avg_model_token_loss,
                        stderr_model_token_loss,
                    ))
                end
            end
        end
    end
    df = DataFrame(rows)
    return df
end

function avg_molnet_info_loss(df=TokenizerStats.info_loss_stats())
    subset!(df, :dataset => ByRow(!=("realspace")))
    subset!(df, :dataset => ByRow(!=("tmQM")))
    @assert "split" ∉ names(df) "Expecting info loss to only be for the val split"
    return combine(groupby(df, [:tokenizer, :ref_tokenizer, :ngram])) do gdf
        # Information Loss Statistics
        avg_info_loss = mean(gdf.avg_info_loss, Weights(gdf.samples))
        nonzero = subset(gdf, :n_nonzero => ByRow(x -> x > 0))
        avg_nonzero_info_loss = mean(nonzero.avg_info_loss, Weights(nonzero.n_nonzero))
        n_nonzero = sum(nonzero.n_nonzero)
        samples = sum(gdf.samples)
        return (;
            vocab_size=first(gdf.vocab_size),
            avg_info_loss,
            avg_perplexity=exp(avg_info_loss),
            samples,
            avg_nonzero_info_loss,
            nonzero_perplexity=exp(avg_nonzero_info_loss),
            n_nonzero,
            nonzero_rate=n_nonzero / samples,
        )
    end
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

function oov_stats()
    data = JSON.parsefile(joinpath(@__DIR__, "..", "stats-atomic.json"))
    rows = []
    for (tokenizer, tok_data) in pairs(data)
        for (dataset, stats) in pairs(tok_data)
            ds_name = startswith(dataset, "MoleculeNet") ? lowercase(split(dataset, "/")[end]) : dataset
            ds_name = replace(ds_name, "lipophilicity" => "lipo")
            push!(rows, (;
                tokenizer,
                molnet=startswith(dataset, "MoleculeNet"),
                dataset=ds_name,
                samples=stats["nobs"],
                n_oov=stats["oov"],
                oov_samples=stats["oov_samples"],
            ))
        end
    end
    return DataFrame(rows)
end

function tokenizer_summary(; k=5)
    tokenizers = JSON.parsefile(joinpath(@__DIR__, "..", "tokenizers.json"))
    stats_dir = joinpath(@__DIR__, "..", "stats")
    smirk = load_tokenizer("smirk")
    rows = []
    for tokenizer in tokenizers
        @info tokenizer
        occursin("smirk-gpe", tokenizer["name"]) && continue
        tok = load_tokenizer(tokenizer["name_or_path"])
        domain = tokenizer["domain"]

        # Count carbon containing tokens
        local n_carbon_tokens
        local carbon_tokens
        if domain != :nlp
            vocab = pyconvert(Vector{String}, values(tok.get_vocab()))
            carbon_tokens = filter(x -> has_element(smirk, x), vocab)
            n_carbon_tokens = length(carbon_tokens)
        else
            carbon_tokens = missing
            n_carbon_tokens = missing
        end

        # Get top-5 tokens
        usage_file = joinpath(stats_dir, tokenizer["name_or_path"], "realspace", "usage.jld2")
        top_tokens = isfile(usage_file) ? top_k_tokens(usage_file; k) : missing

        push!(rows, (;
            tokenizer=tokenizer["name_or_path"],
            domain=tokenizer["domain"],
            class=tokenizer["tokenizer_class"],
            encoding=tokenizer["encoding"],
            vocab_size=pyconvert(Int, length(tok)),
            carbon_tokens,
            n_carbon_tokens,
            top_tokens
        ))
    end
    df = DataFrame(rows)
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
        gram = tokens[i:min(i + 1, length(tokens))]
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

function top_k_tokens(ngram_file; k=5)
    # Load ngram model
    ngram, tok, _ = load_ngram_model(ngram_file)

    # Find the top k tokens
    id_count = collect(pairs(ngram.ngrams[1]))
    sort!(id_count; rev=true, by=last)
    top_k = [first(id) for (id, _) in id_count[1:k]]

    # Get the tokens
    try
        return pyconvert(Vector{String}, map(tok.decode, top_k))
    catch
        @warn "failed to decode tokens for $ngram_file"
        return missing
    end
end
