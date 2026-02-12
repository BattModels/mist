function figure_jaccard(stats_dir)
    # Compare the vocabularies of different tokenizers
    tokenizers = tokenizers_info(stats_dir)
    J, names = tokenizer_jaccard(stats_dir, collect(keys(tokenizers)))
    names = [tokenizers[name_or_path]["name"] for name_or_path in names]
    figure_jaccard(J, names)
end

function figure_jaccard(J::Matrix, names::Vector{String})
    # Cluster hierarchically
    h = hclust(J; linkage=:single, branchorder=:optimal)
    J = J[h.order, h.order]
    names = names[h.order]

    # Reports stats
    @info "Jacard Index 90th percentile" quantile(filter(!=(1), vec(J)), 0.90)

    f = Figure(;
        size=72 .* (6, 5),
        figure_padding=(1, 1, 1, 5),
    )
    ax = Axis(f[1, 1];
        aspect=1,
        xticks=(1:length(names), names),
        yticks=(1:length(names), names),
        xticklabelrotation=pi / 4,
        xticklabelsize=6,
        yticklabelsize=6,
    )
    h = heatmap!(ax, J; colormap=:lipari, colorrange=(0, 1))
    Colorbar(f[1, 2];
        tickformat="{:.0%}",
        ticks=LinearTicks(6),
        label="Jaccard Index",
        colormap=h.colormap,
        colorrange=h.colorrange,
    )
    colgap!(f.layout, 5)
    resize_to_layout!(f)
    return f
end

function tokenizer_jaccard(stats_dir::String, tokenizers::Vector{String})
    tok_tokens = map(tokenizers) do tokenizer
        name_or_path = TokenizerStats.resolve_tok_path(stats_dir, tokenizer)
        tok = TokenizerStats.load_tokenizer(name_or_path)
        vocab_size = pyconvert(Int, length(tok))
        pytokens = tok.convert_ids_to_tokens(collect(range(0; length=vocab_size)))
        tokens = map(token -> pyconvert(String, token, nothing), pytokens)
        return tokenizer => filter(!isnothing, tokens)
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
