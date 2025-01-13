function figure_ngram_prediction(smi, stats_dir; direction=:forward)
    f = Figure(size=(3.42inch, 3inch))
    cb = Colorbar(f[1:3, 4];
        label="Log Probability",
        colormap=:lipari,
        colorrange=(-10, 0),
        tickformat="{:2d}",
    )

    tokenizers = tokenizers_info(stats_dir)
    path(name_or_path) = (joinpath(stats_dir, name_or_path, "realspace", "usage.jld2"), tokenizers[name_or_path]["name"])
    tok_log_prob!(f[1, 1], cb, path("smirk")..., smi; direction)
    tok_log_prob!(f[1, 2], cb, path("ibm/MoLFormer-XL-both-10pct-oov")..., smi; direction)
    tok_log_prob!(f[1, 3], cb, path("seyonec/ChemBERTa-zinc-base-v1")..., smi; direction)
    tok_log_prob!(f[2, 1], cb, path("devalab/molgpt-moses")..., smi; direction)
    tok_log_prob!(f[2, 2], cb, path("rxn4chemistry/rxnfp")..., smi; direction)
    tok_log_prob!(f[2, 3], cb, path("MolecularAI/Chemformer")..., smi; direction)
    tok_log_prob!(f[3, 1], cb, path("meta-llama/Meta-Llama-3.1-8B")..., smi; direction)
    tok_log_prob!(f[3, 2], cb, path("Xenova/gpt-4o")..., smi; direction)
    tok_log_prob!(f[3, 3], cb, path("google/gemma-7b")..., smi; direction)

    # Format plot
    Label(f[:, 0], smi, rotation=pi / 2, fontsize=length(smi) > 40 ? 6 : 8, padding=(0, 2, 0, 0))
    Label(f[end+1, :], "Predicted Tokens", fontsize=8)
    resize_to_layout!(f)
    rowgap!(f.layout, 2)
    colgap!(f.layout, 1)

    return f
end

function tok_log_prob!(f, cb, file, name, smi, max_vocab=50; direction=:forward)
    ngram, tok, info = TokenizerStats.load_ngram_model(file)
    code = pyconvert(Vector{Int}, tok(smi)["input_ids"])

    if direction == :forward
        P = TokenizerStats.autoregressive_log_prob(ngram, code)
    elseif direction == :bidirectional
        P = TokenizerStats.fb_log_probability(ngram, code)
    else
        error("unknown type $type")
    end
    l = TokenizerStats.cross_entropy(P, code)

    vocab = TokenizerStats.nonspecial_vocab(ngram)
    P = P[vocab.+1, :]

    # Truncate Vocab
    max_vocab = min(max_vocab, size(P, 1))
    name = size(P, 1) > max_vocab ? "*" * name : name
    sdx = sortperm(vec(sum(P; dims=2)); rev=true)
    P = P[sdx[1:max_vocab], :]

    ax = Axis(f;
        title="$(name): $(round(l; sigdigits=2))",
        limits=((0, size(P, 1)), (0, size(P, 2))),
        titlegap=1,
        xticksvisible=false,
        xticklabelsvisible=false,
        yticksvisible=false,
        yticklabelsvisible=false,
        spinewidth=0.5,
        aspect=1,
    )
    image!(ax, P; colormap=cb.colormap, colorrange=cb.colorrange, interpolate=false)

    return nothing
end

function figure_ngram_info_loss(;
    smi="C(=Cc1ccccc1)C1=[O+][Cu-3]2([O+]=C(C=Cc3ccccc3)CC(c3ccccc3)=[O+]2)[O+]=C(c2ccccc2)C1",
    token_colors=("[O+]" => :turquoise, "[Cu-3]" => :magenta)
)
    f = Figure(size=72 .* (4.5, 1.7),
        figure_padding=(1, 1, 5, 1),
    )
    cb = Colorbar(f[1, 4];
        label="Log Odds Ratio",
        colormap=:vik,
        colorrange=(-50, 50),
        tellheight=true,
    )

    rsmi, token_color = rich_smi(smi, token_colors...)

    # Load model
    ref_file = joinpath(@__DIR__, "stats", "character", "realspace/usage.jld2")
    ngram, ref_tok, ref_info = TokenizerStats.load_ngram_model(ref_file)
    ref_code = pyconvert(Vector{Int}, ref_tok(smi)["input_ids"])
    kwargs = (; ngram, ref_tok, ref_code, token_color)

    tok_info_loss!(f[1, 1], cb, "smirk", smi; kwargs...)
    tok_info_loss!(f[1, 2], cb, "ibm/MoLFormer-XL-both-10pct-oov", smi; kwargs...)
    tok_info_loss!(f[1, 2], cb, "SmilesPE/SPE_ChEMBL", smi; kwargs...)
    tok_info_loss!(f[2, 1], cb, "MolecularAI/Chemformer", smi; kwargs...)
    tok_info_loss!(f[1, 3], cb, "devalab/molgpt-moses", smi; kwargs...)
    tok_info_loss!(f[2, 3], cb, "rxn4chemistry/rxn_yields", smi; kwargs...)

    # Show vocab
    vocab = TokenizerStats.nonspecial_vocab(ngram)
    unk = pyconvert(Int, ref_tok.unk_token_id)
    filter!(!=(unk), vocab)
    rvocab = Makie.RichText[]
    for (i, id) in enumerate(vocab)
        token = pyconvert(String, ref_tok.decode(id))
        pad = i % 5 == 0 ? "  " : ""
        push!(rvocab, rich(token * pad))
    end

    # Format plot
    Label(f[0, :], rsmi, fontsize=6)
    # Label(f[end+1, :], rich(rvocab...), fontsize=8)
    resize_to_layout!(f)
    colgap!(f.layout, 3)
    rowgap!(f.layout, 3)

    return f
end

function rich_smi(smi::String, colors::Pair...; colormap=:viridis)
    out = []
    colors = Dict(colors)
    idx = firstindex(smi)
    matches = findall(r"\[.*?\]", smi)
    cmap = cgrad(colormap, length(matches))
    token_color = Vector(undef, length(smi))
    token_color .= :black
    for m in matches
        if idx != prevind(smi, first(m))
            r = idx:prevind(smi, first(m))
            token_color[r] .= :black
            push!(out, rich(smi[r]))
        end

        # Get color for segment
        segment = get(colors, smi[m], cmap[length(colors)+1])
        colors[smi[m]] = segment
        token_color[m] .= segment
        push!(out, rich(smi[m]; color=segment, font=:bold))
        idx = nextind(smi, last(m))
    end

    if idx != lastindex(smi)
        push!(out, rich(smi[idx:end]))
        token_color[idx:end] .= :black
    end
    return rich(out...), token_color
end

function box_token!(ax, token_id::Int, code_pos::Int; offset=0.0, kwargs...)
    token_id += 1
    point = [
        (token_id, code_pos),
        (token_id, code_pos + 1),
        (token_id + 1, code_pos + 1),
        (token_id + 1, code_pos),
        (token_id, code_pos),
    ]
    point = map(x -> (x[1] - offset, x[2] - offset), point)
    lines!(ax, point; color=:red, linewidth=0.1, kwargs...)
end

function tok_info_loss!(f, cb, tok::String, smi::String; ngram, ref_tok, ref_code, token_color=missing)

    # Load model
    name = tokenizers_info()[tok]["name"]
    tok = TokenizerStats.load_tokenizer(tok)
    code = pyconvert(Vector{Int}, tok(smi)["input_ids"])

    # Align both tokenizations
    smi_tokens = pyconvert(Vector{String}, tok.tokenize(smi))
    ref_tokens = pyconvert(Vector{String}, ref_tok.tokenize(smi))
    A = TokenizerStats.align_unknown(ref_tokens, smi_tokens)

    # Check for bos/eos tokens
    if length(code) - length(smi_tokens) == 2
        code = code[2:end-1]
    end
    # @assert length(code) == length(smi_tokens)

    # Compute information_loss from unknown tokens
    masked = map(!, vec(any(A; dims=2)))
    @assert length(masked) == length(ref_code)
    i = TokenizerStats.information_loss(ngram, ref_code, masked; N=2)

    # Remove special tokens
    vocab = TokenizerStats.nonspecial_vocab(ngram)
    unk = pyconvert(Int, ref_tok.unk_token_id)
    filter!(!=(unk), vocab)
    P = P[vocab.+1, :]
    Q = Q[vocab.+1, :]

    odds_ratio = @. (Q - log(1 - exp(Q))) - (P - log(1 - exp(P)))
    @info "masked for $name" masked_tokens = join(ref_tokens[masked], " ") extrema(odds_ratio)

    ax = Axis(f;
        title="$name: $(round(i; sigdigits=3))",
        xticks=collect(5:5:length(vocab)),
        xticksvisible=false,
        xticklabelsvisible=false,
        yticksvisible=false,
        yticklabelsvisible=false,
        aspect=1,
        spinewidth=0.5,)
    image!(ax, odds_ratio;
        colormap=cb.colormap,
        colorrange=cb.colorrange,
        highclip=cb.highclip,
        lowclip=cb.lowclip,
        interpolate=false,
    )

    # Highlight the correct token
    @info token_color
    for (code_pos, token_id) in enumerate(ref_code)
        color = ismissing(token_color) ? :black : token_color[code_pos]
        box_token!(ax, token_id, code_pos; linewidth=0.5, color)
    end

    return nothing
end


function figure_kl_v_info_loss(stats_dir, model_loss, info_loss; reference="character")
    tokenizers = tokenizers_info(stats_dir)

    model_loss = subset(model_loss,
        :dataset => ByRow(∉(["realspace", "tmqm"])),
        :split => ByRow(==("val")),
    )
    model_loss = combine(groupby(model_loss, [:tokenizer, :split, :ngram])) do gdf
        loss_per_token_moments = reduce(merge, gdf.loss_per_token_moments)
        loss_moments = reduce(merge, gdf.loss_moments)
        return (;
            avg_model_loss=mean(loss_moments),
            stderr_model_loss=std(loss_moments) / sqrt(nobs(loss_moments)),
            avg_model_token_loss=mean(loss_per_token_moments),
            stderr_model_token_loss=std(loss_per_token_moments) / sqrt(nobs(loss_per_token_moments)),
            vocab_size=first(gdf.vocab_size),
        )
    end

    # Summarize over MoleculeNet
    info_loss = subset(info_loss,
        :dataset => ByRow(∉(["realspace", "tmqm"])),
        :ref_tokenizer => ByRow(==(reference)),
    )
    info_loss = combine(groupby(info_loss, [:tokenizer, :ref_tokenizer, :ngram])) do gdf
        info_loss_moments = reduce(merge, gdf.info_loss_moments)
        return (;
            avg_info_loss=mean(info_loss_moments),
            stderr_info_loss=std(info_loss_moments) / sqrt(nobs(info_loss_moments)),
        )
    end

    df = leftjoin(model_loss, info_loss, on=[:tokenizer, :ngram])
    plt_tokenizers = [
        "smirk-gpe-50k-nmb-ss" => :star8,
        "smirk" => :star5,
        "ibm/MoLFormer-XL-both-10pct-oov" => :diamond,
        "devalab/molgpt-moses" => :ltriangle,
        "devalab/molgpt-guacamol" => :rtriangle,
        # "rxn4chemistry/rxn_yields" => :cross,
        "rxn4chemistry/rxnfp" => :x,
        "sagawa/ReactionT5-product-prediction" => :circle,
        "ChangwenXu98/TransPolymer" => :dtriangle,
        "MolecularAI/Chemformer" => :utriangle,
        "SmilesPE/SPE_ChEMBL" => :hexagon,
    ]
    dropmissing!(df)
    subset!(df,
        :ngram => ByRow(>=(4)),
        :split => ByRow(==("val")),
    )
    @info "Tokens with no info_loss" unique(df.tokenizer)
    # subset!(df, :tokenizer => ByRow(x -> x in first.(plt_tokenizers)))
    df.tokenizer = categorical(df.tokenizer)
    df_sum = select(df, [:tokenizer, :ngram, :avg_model_token_loss, :stderr_model_token_loss, :avg_info_loss, :stderr_info_loss])
    sort!(df_sum, :avg_info_loss; rev=true)
    display(df_sum)

    f = Figure(; size=72 .* (4.5, 3), figure_padding=(1, 1, 1, 4))
    ax = Axis(f[1, 1];
        limits=(nothing, (-0.1, nothing)),
        xlabel="Cross Entropy Loss [nats/token]",
        ylabel="Information Loss [nats]",
        yticks=[0, 0.5, 2, 4, 16, 64, 256, 512],
        yscale=Asinh(1),
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(4),
        yminorgridvisible=true,
        xgridvisible=false,
    )

    errorbars!(ax, df.avg_model_token_loss, df.avg_info_loss, 3 .* df.stderr_model_token_loss; direction=:x, color=:black, linewidth=1)
    errorbars!(ax, df.avg_model_token_loss, df.avg_info_loss, 3 .* df.stderr_info_loss; direction=:y, color=:black, linewidth=1)
    h = scatter!(ax, df.avg_model_token_loss, df.avg_info_loss;
        colormap=:Set2_5,
        colorrange=(1, 5),
        color=df.ngram,
        # marker=map(n -> Dict(plt_tokenizers)[n], df.tokenizer),
    )

    # # Build legend
    # ngram_elements = map(2:5) do gdx
    #     PolyElement(color=gdx, colorrange=h.colorrange, colormap=h.colormap)
    # end
    # ngram_labels = ["Bigram", "Trigram", "4-gram", "5-gram"]
    # tokenizer_elements = map(levels(df.tokenizer)) do name
    #     MarkerElement(; marker=Dict(plt_tokenizers)[name], color=:black)
    # end
    # tokenizer_labels = map(levels(df.tokenizer)) do name_or_path
    #     return tokenizers[name_or_path]["name"]
    # end
    # Legend(f[1, 1],
    #     [ngram_elements, tokenizer_elements],
    #     [ngram_labels, tokenizer_labels],
    #     ["n-gram", "Tokenizer"];
    #     tellheight=false,
    #     tellwidth=false,
    #     halign=:right,
    #     valign=:top,
    # )
    # resize_to_layout!(f)

    return f
end

function figure_info_loss_ref_tokenizer(model_loss)
    tokenizers = ("character", "meta-llama/Meta-Llama-3.1-8B")
    df = subset(model_loss,
        :split => ByRow(==(:val)),
        :ngram => ByRow(>(1)),
        :ref_tokenizer => ByRow(in(tokenizers)),
    )
    select!(df, [:tokenizer, :ref_tokenizer, :ngram, :avg_info_loss])
    df = unstack(df, [:tokenizer, :ngram], :ref_tokenizer, :avg_info_loss)
    dropmissing!(df)
    @info "Info Loss" characters = extrema(df.character) llama = extrema(df[!, "meta-llama/Meta-Llama-3.1-8B"])
    @info "Tokenizers" unique(df.tokenizer)
    display(subset(df, :ngram => ByRow(==(5))))

    f = Figure(; size=72 .* (6.5, 3))
    ax = Axis(f[1, 1];
        title="Information Loss from Unknown Tokens [nats]",
        limits=((0, 30), (0, 0.75)),
        xlabel=TOKENIZERS[tokenizers[1]],
        ylabel=TOKENIZERS[tokenizers[2]],
        aspect=1,
        xscale=sqrt,
        yscale=sqrt,
        xminorticksvisible=true,
        xminorticks=IntervalsBetween(10),
        xminorgridvisible=true,
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(10),
        yminorgridvisible=true,
    )
    lines!(ax, range(0, 30; length=100), range(0, 30; length=100); color=:black, linestyle=:dash)
    scatter!(ax, df[!, tokenizers[1]], df[!, tokenizers[2]];
        marker=:x,
        color=df.ngram,
    )

    model_loss = subset(model_loss,
        :dataset => ByRow(!=("realspace_v4_dev")),
        :training_dataset => ByRow(==("realspace_v4_dev")),
        :split => ByRow(==(:val)),
        :tokenizer => ByRow(in(tokenizers)),
    )
    model_loss = combine(groupby(model_loss, [:tokenizer, :ngram])) do gdf
        return (;
            avg_model_loss=mean(gdf.avg_model_loss, Weights(gdf.samples)),
        )
    end
    model_loss = unstack(model_loss, :ngram, :tokenizer, :avg_model_loss)
    dropmissing!(model_loss)

    ax = Axis(f[1, 2];
        title="Cross-Entropy Loss [nats]",
        limits=((0, 225), (0, 225)),
        aspect=1,
        xlabel=TOKENIZERS[tokenizers[1]],
        ylabel=TOKENIZERS[tokenizers[2]],
    )
    hab = ablines!(ax, 0, 1; color=:black, linestyle=:dash, label="Parity")
    h = scatter!(ax, model_loss[!, tokenizers[1]], model_loss[!, tokenizers[2]];
        marker=:x,
        color=df.ngram,
        colorrange=(1, 5),
    )
    ngram_labels = ["Unigram", "Bigram", "Trigram", "4-gram", "5-gram"]
    ngram_elements = map(1:length(ngram_labels)) do gdx
        MarkerElement(; color=gdx, colorrange=h.colorrange, colormap=h.colormap, marker=h.marker)
    end
    Legend(f[1, 2], [hab, ngram_elements...], [hab.label, ngram_labels...];
        tellheight=false,
        tellwidth=false,
        halign=:right,
        valign=:bottom,
        margin=(10, 10, 5, 10),
    )
    Label(f[1, 1, TopLeft()], "a)";
        font=:bold,
        halign=:right,
        padding=(0, 15, 5, 0),
    )
    Label(f[1, 2, TopLeft()], "b)";
        font=:bold,
        halign=:right,
        padding=(0, 15, 5, 0),
    )

    resize_to_layout!(f)
    display(model_loss)

    return f
end
