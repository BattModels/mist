function serialize_usage!(f::JLD2.JLDFile, path::AbstractString, x::OnlineStats.Series)
    f[joinpath(path, "samples")] = nobs(x)
    f[joinpath(path, "out_of_vocab")] = value(x[:out_of_vocab])
    f[joinpath(path, "fertility")] = value(x[:fertility])
    f[joinpath(path, "nunique")] = value(x[:nunique])
    for n in eachindex(x[:ngrams].stats)
        f[joinpath(path, "ngrams", string(n))] = value(x[:ngrams][n])
    end
    return nothing
end

"""
Re-serialize usage stats from the old format to the new format

Primary difference, is stats are broken out into groups allowing for partially
loading the data. Prior format only allowed limiting loads to a single split.
"""
function flatten_usage_stats(filename::AbstractString)
    dst = filename * ".v2"
    jldopen(filename, "r") do f
        jldopen(dst, "w") do o
            o["tokenizer"] = f["tokenizer"]
            for split in ["val", "train", "test"]
                haskey(f, split) || continue
                o[joinpath(split, "samples")] = f[split].samples
                o[joinpath(split, "out_of_vocab")] = f[split].out_of_vocab
                o[joinpath(split, "fertility")] = Dict(f[split].fertility)
                for n in eachindex(f[split].ngrams)
                    o[joinpath(split, "ngrams", string(n))] = Dict(f[split].ngrams[n])
                end
            end
        end
    end
    return dst
end

function correct_model_loss(filename::AbstractString)
    dst = filename * ".fix_cross_entropy"
    jldopen(filename, "r") do f
        jldopen(dst, "w") do o
            o["tokenizer"] = f["tokenizer"]
            o["ref_tokenizer"] = f["ref_tokenizer"]
            for split in ["val", "train", "test"]
                data = f[split]
                o[split] = (; samples=data.samples, cross_entropy=data.kld, cross_entropy_per_token=data.kld_per_token)
            end
        end
    end
    return dst
end
