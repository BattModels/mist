function usage_stats(stats_dir)
    rows = []
    for file in find(stats_dir, r"usage.jld2$")
        jldopen(file) do data
            file = relpath(file, stats_dir)
            tokenizer = joinpath(splitpath(file)[1:end-2])
            dataset = splitpath(file)[end-1]
            dataset = dataset == "tmqm" ? "tmQM" : dataset
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
    for file in find(stats_dir, r"/model_loss\.jld2$")

        # Get average fertility
        usage = joinpath(dirname(file), "usage.jld2")
        if !isfile(usage)
            @warn "No usage file found for $file"
            avg_fertility = Dict{String,Float64}()
        else
            avg_fertility = jldopen(usage, "r") do data
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
            dataset = dataset == "tmqm" ? "tmQM" : dataset
            for split in ["train", "val", "test"]
                split ∉ keys(data) && continue
                split_data = data[split]
                for ngram in 1:5
                    cross_entropy = OnlineStats.Moments(split_data[:cross_entropy][ngram][:moments], EqualWeight(), split_data[:samples])
                    cross_entropy_per_token = OnlineStats.Moments(split_data[:cross_entropy_per_token][ngram][:moments], EqualWeight(), split_data[:samples])

                    push!(rows, (;
                        tokenizer,
                        dataset,
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

function tokenizer_summary(stats_dir; k=5)
    tokenizers = JSON.parsefile(joinpath(stats_dir, "tokenizers.json"))
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
    ngram, tok, _ = TokenizerStats.load_ngram_model(ngram_file)

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
