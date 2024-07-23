using Makie
using CairoMakie
using PythonCall
using JSON: JSON

plot_parity!(ax, filename::String; kwargs...) = plot_parity!(ax, JSON.parsefile(filename); kwargs...)

function plot_parity!(ax, data::Dict; kwargs...)
    parity = data["parity"]
    molecules = data["exact_count"]
    n_tokens = collect(parse.(Int, keys(parity)))
    freq = 100 .* collect(values(parity) ./ molecules)
    barplot!(ax,
        n_tokens, freq;
        width=1,
        gap=0,
        kwargs...
    )
end

function plot_rank_freq!(ax, tokenizer, data::Dict; kwargs...)
    token_counts = data["token_counts"]
    token_ids = collect(parse.(Int, keys(token_counts)))
    token_freq = collect(Float64, values(token_counts))
    # token_freq ./= sum(token_freq)
    sdx = sortperm(token_freq; rev=true)
    permute!(token_freq, sdx)
    permute!(token_ids, sdx)
    vocab_size = pyconvert(Int, tokenizer.vocab_size)
    append!(token_freq, fill(0, vocab_size - length(token_freq)))
    lines!(ax, range(1, vocab_size), token_freq; kwargs...)
end

function plot(data:: Dict{String, String})
    f = Figure(size=(600, 300))

    # Plot Tokenizer Parity
    ax_parity = Axis(f[1, 1];
        xlabel="Number of Tokens",
        ylabel="Fraction of Molecules",
        ytickformat="{:.0f}%",
        limits=((1, nothing), nothing),
    )

    # Plot Tokenizer Rank-Freqency
    ax_rank = Axis(f[1, 2];
        xlabel="Rank",
        ylabel="Freqency",
        yscale=log10,
        xscale=log10,
        limits=((1, nothing), (1, 1e9)),
    )

    AutoTokenizer = pyimport("transformers").AutoTokenizer
    smirk = pyimport("smirk")

    for (label, file) in data
        stats = JSON.parsefile(file)
        tokenizer_name = basename(dirname(file))
        if startswith(tokenizer_name, "smirk")
            tokenizer = smirk.SmirkTokenizerFast.from_pretrained(dirname(file))
        else
            tokenizer = AutoTokenizer.from_pretrained(dirname(file), cache_dir=".cache", trust_remote_code=true)
        end
        @info "Tokenizer" tokenizer.vocab_size
        plot_parity!(ax_parity, stats)
        plot_rank_freq!(ax_rank, tokenizer, stats; label)
    end
    axislegend(ax_rank; position=:rt)


    save(joinpath(@__DIR__, "tokenizer_stats.pdf"), f)
    f
end
