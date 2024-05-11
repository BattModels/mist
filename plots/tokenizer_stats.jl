using Makie
using CairoMakie
using JSON: JSON

plot_parity!(ax, filename:: String; kwargs...) = plot_parity!(ax, JSON.parsefile(filename); kwargs...)

function plot_parity!(ax, data:: Dict; kwargs...)
    parity = data["parity"]
    molecules = data["exact_count"]
    n_tokens = collect(parse.(Int, keys(parity)))
    freq = collect(values(parity) ./ molecules)
    barplot!(ax,
        n_tokens, freq;
        width=1,
        gap=0,
        kwargs...
    )
end

function plot()
    f = Figure()
    ax_parity = Axis(f[1, 1]; xlabel="Number of Tokens", ylabel="Fraction of Molecules")
    plot_parity!(ax_parity, "smirk-piece-256/stats-15505052.json")
    save(joinpath(@__DIR__, "tokenizer_stats.pdf"), f)
end
