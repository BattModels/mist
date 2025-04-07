using Makie
using ManifoldLearning
using SafeTensors: SafeTensors
using JSON: JSON
using Glob: @fn_str
using EnumX: @enumx
using MISTStyle
using LinearAlgebra: norm, dot
using Statistics: mean, std


function token_embedding(model::String)
    # Get Token embeddings
    tensors = SafeTensors.deserialize(joinpath(model, "model.safetensors"))
    key = first(filter(x -> occursin(fn"*.word_embeddings.weight", x), keys(tensors)))
    emb = tensors[key]

    # Get Token id => symbol mapping
    tok_config = JSON.parsefile(joinpath(model, "tokenizer.json"))

    vocab = tok_config["model"]["vocab"]
    added_tokens = Dict(tok["content"] => tok["id"] for tok in tok_config["added_tokens"])
    merge!(vocab, added_tokens)
    @assert size(emb, 1) == length(vocab) "Missing tokens in vocab"
    @assert extrema(values(vocab)) == (0, size(emb, 1) - 1) "Holes in token id space"
    sdx = sortperm(collect(values(vocab)))
    tokens = collect(keys(vocab))[sdx]
    return emb, tokens
end

@enumx TokenType begin
    Element
    Bond
    Structure
    Chiral
    Numeric
    Other
    Special
end

function token_type(token::String)
    if occursin(r"^[A-Za-z][a-z]?", token)
        return TokenType.Element
    elseif token in ["[", "]", "(", ")", "/", "\\", "%"]
        return TokenType.Structure
    elseif token in ["=", "#", "\$", ".", ":"]
        return TokenType.Bond
    elseif occursin(r"\[[A-Z]*]]", token)
        return TokenType.Special
    elseif token in ["@", "@@"] || occursin(r"[A-Z]{2}", token)
        return TokenType.Chiral
    elseif occursin(r"[\d\-\+]", token)
        return TokenType.Numeric
    else
        return TokenType.Other
    end
end

cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))

zscore(x; kwargs...) = (x .- mean(x; kwargs...)) ./ std(x; kwargs...)
center(x; kwargs...) = (x .- minimum(x; kwargs...)) ./ (maximum(x; kwargs...) .- minimum(x; kwargs...))

function figure_token_embeddings(models; last_token=75, emb_models=nothing, fig_size=(3.42inch, 1.5inch), min_update=0.005)
    # Token movement during finetuning
    ref_emb, ref_tokens = token_embedding(models[1][2])
    emb_movement = []
    for model in last.((models[2:end]))
        emb, m_toks = token_embedding(model)
        @assert all(m_toks .== ref_tokens) "Tokens don't match"
        d = 1 .- map(cosine_similarity, eachrow(ref_emb), eachrow(emb))
        push!(emb_movement, d)
    end
    emb_movement = reduce(hcat, emb_movement)
    @assert 0 <= minimum(emb_movement) && maximum(emb_movement) <= 1 "Unexpected Cos. Dist. Range"
    avg_update = vec(mean(emb_movement, dims=2))
    sdx = sortperm(avg_update; rev=true)
    emb_movement = emb_movement[sdx, :]
    avg_update = avg_update[sdx]
    tokens = ref_tokens[sdx]

    # Order models by average token movement
    sdx = sortperm(vec(mean(emb_movement; dims=1)); rev=true)
    emb_movement = emb_movement[:, sdx]
    model_labels = first.(models[2:end])[sdx]

    # Restrict to tokens with a significant movement
    @info "tokens with dist > $min_update" searchsortedlast(avg_update, min_update, rev=true)
    last_token = Int(min(last_token, length(tokens)))
    emb_movement = emb_movement[1:last_token, :]
    tokens = tokens[1:last_token]

    # Group tokens by type
    tok_class = token_type.(tokens)
    sdx = sortperm(tok_class)
    emb_movement = emb_movement[sdx, :]
    tokens = tokens[sdx]

    # Plot Results
    f = Figure(;
        size=fig_size,
        figure_padding=(2, 2, 2, 3)
    )
    gl = GridLayout(f[1, 1])
    MISTStyle.sublabel!(f[1, 1, TopLeft()], "a"; left=15)
    ax = Axis(gl[1, 1];
        xticks=(axes(emb_movement, 1), tokens),
        xticklabelrotation=pi / 2,
        xticklabelsize=5pt,
        xticksvisible=false,
        yticksvisible=false,
        yticks=(axes(emb_movement, 2), model_labels),
    )
    h = heatmap!(ax, axes(emb_movement)..., emb_movement;
        colorrange=(min_update, 1),
        colorscale=log10,)
    Colorbar(gl[1, 2], h;
        ticks=LogTicks(WilkinsonTicks(3)),
        label="Cosine Distance",
        size=4pt,
    )

    gl = GridLayout(f[2, 1])
    MISTStyle.sublabel!(f[2, 1, TopLeft()], "b"; left=5)
    ax_per_row = 4
    emb_models = isnothing(emb_models) ? length(models) : emb_models
    for (idx, (label, model)) in enumerate(models[1:emb_models])
        emb, mtoks = token_embedding(model)
        emb = zscore(Float64.(emb); dims=2)
        r = predict(fit(TSNE, emb'))
        rdx = fld(idx - 1, ax_per_row) + 1
        cdx = (idx - 1) % ax_per_row + 1
        ax = Axis(gl[rdx, cdx]; title=label)
        hidedecorations!(ax)
        for (tok, pos) in zip(mtoks, eachcol(r))
            tok in tokens || continue
            text!(ax, pos[1], pos[2];
                text=tok,
                color=Int(token_type(tok)),
                colormap=MISTStyle.CAT_COLORS,
                colorrange=(0, 9),
            )
        end
    end
    emb_models > 4 && rowsize!(f.layout, 1, 1inch)
    resize_to_layout!(f)
    return f
end
