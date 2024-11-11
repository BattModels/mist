using TokenizerStats
using JLD2
using Serialization

# Export to JLD2
data = Dict()
for split in ["train", "val", "test"]
    if !isfile("/tmp/" * split * ".jld2")
        @info "merging" split
        split_data = TokenizerStats.merge_usage_stats("/tmp/SmilesPE/"; split)
        for (n, ngram) in enumerate(split_data[Symbol(split)].ngrams)
            @info "ngram-$n" length(ngram) typeof(ngram)
        end
        data[split] = split_data
        @info "saving" split
        serialize("/tmp/" * split * ".jld", split_data)
    end
end

@info "merging in memory"
out = (;
    tokenizer = data["train"][:tokenizer],
    train = data["train"][:train],
    val = data["val"][:val],
    test = data["test"][:test],
)
serialize("merged.jld", out)

