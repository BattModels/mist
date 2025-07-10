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
    Hydrogen
    Element
    Bond
    Structure
    Chiral
    Numeric
    Charge
    Aromatic
    Halogen
    Chalcogen
    Pnictogen
    Tetrel
    Triel
    dBlock
    fBlock
    Other
    Special
end

function token_type(token::String)
    if token == "H"
        return TokenType.Hydrogen
    elseif token in ["F", "Cl", "Br", "I", "At", "Ts"]
        return TokenType.Halogen
    elseif token in ["O", "S", "Se", "Te", "Po", "Lv"]
        return TokenType.Chalcogen
    elseif token in ["N", "P", "As", "Sb", "Bi", "Mc"]
        return TokenType.Pnictogen
    elseif token in ["C", "Si", "Ge", "Sn", "Pb", "Fl"]
        return TokenType.Tetrel
    elseif token in ["B", "Al", "Ga", "In", "Tl", "Nh"]
        return TokenType.Triel
    elseif token in [
        # Lanthanides
        "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
        "Ho", "Er", "Tm", "Yb", "Lu",
        # Actinides
        "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
        "Es", "Fm", "Md", "No", "Lr"
    ]
        return TokenType.fBlock
    elseif all(islowercase, token)
        return TokenType.Aromatic
    elseif token in [
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    ]
        return TokenType.dBlock
    elseif occursin(r"^[A-Za-z][a-z]?", token)
        return TokenType.Element
    elseif token in ["[", "]", "(", ")", "/", "\\", "%"]
        return TokenType.Structure
    elseif token in ["=", "#", "\$", ".", ":"]
        return TokenType.Bond
    elseif occursin(r"^\[[A-Z]*]$", token)
        return TokenType.Special
    elseif token in ["@", "@@"] || occursin(r"^@[A-Z]{2}$", token)
        return TokenType.Chiral
    elseif occursin(r"[\d]", token)
        return TokenType.Numeric
    elseif token in ["+", "-"]
        return TokenType.Charge
    else
        return TokenType.Other
    end
end

cosine_similarity(a, b) = dot(a, b) / (norm(a) * norm(b))

zscore(x; kwargs...) = (x .- mean(x; kwargs...)) ./ std(x; kwargs...)
center(x; kwargs...) = (x .- minimum(x; kwargs...)) ./ (maximum(x; kwargs...) .- minimum(x; kwargs...))

function token_colormap()
    colors = collect(cgrad(:glasbey_bw_minc_20_n256))
    popat!(colors, 5)
    popat!(colors, 6)
    popat!(colors, 10)
    popat!(colors, 11)
    popat!(colors, 11)
    popat!(colors, 15)
    return colors
end
function token_ticks(tokens::Vector{String})
    colormap = token_colormap()
    labels = map(tokens) do token
        tt = token_type(token)
        color = colormap[Int(tt)+1]
        rich(token; color, font=:bold)
    end
    return (collect(eachindex(labels)), labels)
end

function eval_token_embeds(models; last_token=75, min_update=0.005)
    # Token movement during finetuning
    ref_emb, ref_tokens = token_embedding(models[1][2])
    vecl2 = x -> norm.(eachrow(x))
    emb_distance = [vecl2(ref_emb)]
    emb_movement = []
    for model in last.((models[2:end]))
        emb, m_toks = token_embedding(model)
        @assert all(m_toks .== ref_tokens) "Tokens don't match"
        d = 1 .- map(cosine_similarity, eachrow(ref_emb), eachrow(emb))
        push!(emb_movement, d)
        push!(emb_distance, vecl2(emb))
    end
    emb_movement = reduce(hcat, emb_movement)
    emb_distance = reduce(hcat, emb_distance)
    @assert 0 <= minimum(emb_movement) && maximum(emb_movement) <= 1 "Unexpected Cos. Dist. Range"
    avg_update = vec(median(emb_movement, dims=2))
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

    return tokens, emb_movement, zip(model_labels, emb_distance)
end

function plot_token_tsne!(f, models; ax_per_row=4, tokens=nothing, colormap=token_colormap())
    for (idx, (label, model)) in enumerate(models)
        emb, mtoks = token_embedding(model)
        emb = zscore(Float64.(emb); dims=2)
        r = predict(fit(TSNE, emb'))
        rdx = fld(idx - 1, ax_per_row) + 1
        cdx = (idx - 1) % ax_per_row + 1
        ax = Axis(
            f[rdx, cdx];
            title=label,
            limits=(nothing, nothing),
        )
        hidedecorations!(ax)
        for (tok, pos) in zip(mtoks, eachcol(r))
            isnothing(tokens) || tok in tokens || continue
            tok in ["[MASK]", "[CLS]", "[UNK]", "[PAD]"] && continue
            text!(ax, pos[1], pos[2];
                text=tok,
                color=Int(token_type(tok)),
                colormap,
                colorrange=(0, length(colormap)-1),
                align=(:center, :center),
            )
        end
    end
    return f
end

function figure_token_embeddings(models; last_token=75, emb_models=nothing, fig_size=(4.5inch, 2inch), min_update=0.005)
    f = Figure(;
        size=(360, 115),
        figure_padding=(2, 2, 2, 3)
    )
    tokens, emb_movement, emb_distance = eval_token_embeds(models; last_token, min_update)

    # Token Movement
    model_labels = first.(emb_distance)
    ax = Axis(f[1, 2];
        yticks=token_ticks(tokens),
        xticklabelrotation=0.3,
        xticklabelsize=5pt,
        xticksvisible=true,
        yticksvisible=false,
        xticks=(axes(emb_movement, 2), model_labels),
        yticklabelsize=5pt,
    )
    h = heatmap!(ax, emb_movement';
        colorrange=(min_update, 1),
        colorscale=log10,)
    Colorbar(f[1, 3], h;
        ticks=LogTicks(WilkinsonTicks(3)),
        label="Cosine Distance from Pretrained",
        size=5pt,
    )

    gl = GridLayout(f[1, 1])
    emb_models = isnothing(emb_models) ? length(models) : emb_models
    colormap = token_colormap()

    tt_legend = map(unique(token_type.(tokens))) do tt
        color = colormap[Int(tt)+1]
        if tt == TokenType.dBlock
            label = "D-Block"
        elseif tt == TokenType.fBlock
            label = "F-Block"
        elseif tt == TokenType.Element
            label = "Elements"
        elseif tt == TokenType.Numeric
            label = "Digit"
        else
            label = string(tt)
        end
        return PolyElement(; color, label)
    end
    Legend(gl[2, 1], [tt_legend], [label.(tt_legend)], ["Token Class"];
        titleposition=:left,
        titlegap=5pt,
        orientation=:horizontal,
        tellheight=true,
        tellwidth=false,
        nbanks=2,
    )
    glt = GridLayout(gl[1,1])
    plot_token_tsne!(glt, models[1:emb_models]; colormap)


    rowgap!(f.layout, 3pt)
    colgap!(f.layout, 3pt)
    colsize!(f.layout, 1, Relative(3 / 4))
    resize_to_layout!(f)
    return f
end

function plot_token_clusters(args...; kwargs...)
    f = Figure()
    plot_token_clusters!(f, args...; kwargs...)
    return f
end

function plot_token_clusters!(f, emb, tokens; correlation=cosine_similarity, kwargs...)
    dist = correlation.(eachrow(emb), eachrow(emb)')

    # Order entries by clustering
    c = hclust(dist; linkage=:ward, branchorder=:barjoseph)
    dist = dist[c.order, c.order]
    columns = tokens[c.order]

    ticks = (eachindex(columns), columns)
    ax = Axis(f[1, 1];
        xticks=ticks, yticks=ticks,
        xticklabelrotation=0.55,
        xticklabelsize=5pt,
        yticklabelsize=5pt,
        aspect=DataAspect(),
    )
    h = heatmap!(ax, dist; colorrange=(-1, 1), colormap=:vik)
    cb = Colorbar(f[1, 2], h)
    return h
end
