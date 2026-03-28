#!/usr/bin/env -S julia +release --color=auto --startup-file=no --project=@script
# Estimate the compute cost of doing a brute force search
# Usage:
#   ./brute_force_cost.jl RUNS_JSONL OUTPUT_TEX
# Example:
#   ./brute_force_cost.jl ../out/dec_3_sweep_runs.jsonl cost_summary.tex

using DataFrames
using JSON: JSON
using BayesianScaling: non_embedding_size, pf_day

const AVG_SEQ_LENGTH::Float64 = 65.2
"""
Average number of tokens per molecule for the Smirk Tokenizer on REALSpace.
Source: SI for doi:10.1021/acs.jcim.5c01856
"""

function read_jsonl(file::AbstractString)
    rows = []
    open(file, "r") do fid
        for line in eachline(fid)
            push!(rows, JSON.parse(line))
        end
    end
    return DataFrame(rows)
end

function unique_lr_prefactor(df::DataFrame)
    # Round to avoid 3.125 and 3.1249999... being treated differently.
    lr_prefactor = @. df.lr / sqrt(df.effective_batch_size)
    return unique(round.(lr_prefactor; sigdigits=3))
end

function compute_full_cost(df::DataFrame)
    levels = (;
        d_model = unique(df.d_model),
        n_layer = unique(df.d_model ./ df.aspect_ratio),
        ff_ratio = unique(df.ff_ratio),
        effective_batch_size = unique(df.effective_batch_size),
        kv_size = unique(df.kv_size),
        data_size = unique(df.max_steps .* df.effective_batch_size),
        lr_prefactor = unique_lr_prefactor(df),
    )
    l_names = keys(levels)
    total_cost = sum(Iterators.product(levels...)) do p
        p = NamedTuple{l_names}(p)
        N = non_embedding_size(p.d_model, p.ff_ratio, p.n_layer)
        D = p.data_size
        return 6 * N * D * AVG_SEQ_LENGTH
    end
    return levels, total_cost
end

function compute_partial_cost(df::DataFrame; include_lr::Bool=true)
    levels = (;
        model_size = unique(df.model_size),
        data_size = unique(df.effective_batch_size .* df.max_steps),
    )
    if include_lr
        levels = (; levels..., lr_prefactor = unique_lr_prefactor(df))
    end
    l_names = keys(levels)
    total_cost = sum(Iterators.product(levels...)) do p
        p = NamedTuple{l_names}(p)
        return 6 * p.model_size * p.data_size * AVG_SEQ_LENGTH
    end
    return levels, total_cost
end

function compute_bayes_cost(df::DataFrame)
    N = df.model_size
    D = @. df.effective_batch_size * df.max_steps * AVG_SEQ_LENGTH
    return sum(x -> 6 * prod(x), zip(N, D))
end

function estimate_cost(levels::NamedTuple)
    l_names = keys(levels)
    total_compute = sum(Iterators.product(levels...)) do p
        p = NamedTuple{l_names}(p)
        N = non_embedding_size(p.d_model, p.ff_ratio, p.n_layer)
        D = p.data_size
        return 6 * N * D * AVG_SEQ_LENGTH
    end
    return total_compute
end

function show_levels(levels::NamedTuple)
    for (k, v) in pairs(levels)
        n = length(v)
        println("$k ($n): $v")
    end
    return nothing
end

function round_sig(x::Real; sigdigits::Int=3)
    return round(float(x); sigdigits=sigdigits)
end

function latex_escape_identifier(x)
    s = String(x)
    return replace(s, "_" => "\\_")
end

function latex_math_identifier(x)
    s = String(x)
    if occursin("_", s)
        head, tail... = split(s, "_")
        return "\$" * head * "_{\\text{" * join(tail, "\\_") * "}}\$"
    end
    return "\$" * s * "\$"
end

function latex_inline_identifier(x)
    if x in (:d_model, :n_layer, :kv_size)
        return latex_math_identifier(x)
    elseif x == :data_size
        return "\$D\$"
    elseif x == :lr_prefactor
        return "\$lr\$"
    elseif x == :ff_ratio
        return "ff\\_ratio"
    elseif x == :effective_batch_size
        return "batch"
    else
        return latex_escape_identifier(x)
    end
end

function latex_description_label(x)
    if x in (:d_model, :n_layer, :kv_size)
        return latex_math_identifier(x)
    elseif x == :lr_prefactor
        return "\$lr\\_prefactor\$"
    else
        return latex_escape_identifier(x)
    end
end

function latex_number(x::Real; sigdigits::Int=6)
    xr = round(float(x); sigdigits=sigdigits)

    if iszero(xr)
        return "0"
    end

    if 1e-3 <= abs(xr) < 1e4
        if isinteger(xr)
            return string(Int(round(xr)))
        end
        return string(xr)
    end

    exp10 = floor(Int, log10(abs(xr)))
    mant = xr / 10.0^exp10
    mant = round(mant; sigdigits=sigdigits)
    return "\\sn{$mant}{$exp10}"
end

function latex_vector(values; sigdigits::Int=6)
    parts = [latex_number(v; sigdigits=sigdigits) for v in sort(collect(values))]
    return "[" * join(parts, ",\\;\n") * "]"
end

function latex_levels_block(
    io::IO,
    title::AbstractString,
    levels::NamedTuple;
    skip::Set{Symbol}=Set{Symbol}(),
)
    println(io, "\\paragraph{$title}")
    println(io, "\\begin{description}")
    println(io)

    for (k, v) in pairs(levels)
        if k in skip
            continue
        end
        n = length(v)
        label = latex_description_label(k)
        vec = latex_vector(v)
        println(io, "\\item[$label ($n)]")
        println(io, "\\[")
        println(io, vec)
        println(io, "\\]")
        println(io)
    end

    println(io, "\\end{description}")
    println(io)
    return nothing
end

function generate_cost_latex(
    full_levels::NamedTuple,
    C_full::Real,
    partial_levels::NamedTuple,
    C_partial::Real,
    partial_levels_lr::NamedTuple,
    C_partial_lr::Real,
    C_bayes::Real;
    cost_3lr::Union{Nothing, Real}=nothing,
    sigdigits_cost::Int=3,
)
    rs(x) = round(float(x); sigdigits=sigdigits_cost)

    rel_full = C_full / C_bayes
    rel_partial = C_partial / C_bayes
    rel_partial_3lr = isnothing(cost_3lr) ? nothing : cost_3lr / C_bayes
    rel_partial_lr = C_partial_lr / C_bayes

    full_level_summary = join([
        "$(latex_inline_identifier(k))($(length(v)))" for (k, v) in pairs(full_levels)
    ], ", ")

    partial_level_summary = join([
        "$(latex_inline_identifier(k))($(length(v)))" for (k, v) in pairs(partial_levels)
    ], ", ")

    partial_lr_level_summary = join([
        "$(latex_inline_identifier(k))($(length(v)))" for (k, v) in pairs(partial_levels_lr)
    ], ", ")

    io = IOBuffer()

    println(io, "\\begin{table}[h]")
    println(io, "\\centering")
    println(io, "\\begin{tabular}{lccc}")
    println(io, "\\hline")
    println(io, "Case & Levels (with \$n\$) & Total Cost (PF-days) & Cost Rel.\\ Bayes \\\\")
    println(io, "\\hline")
    println(io, "Full Factorial & $full_level_summary & $(rs(C_full / pf_day)) & $(rs(rel_full)) \\\\")
    println(io, "\$N + D\$ Only & $partial_level_summary & $(rs(C_partial / pf_day)) & $(rs(rel_partial)) \\\\")
    if !isnothing(cost_3lr)
        println(
            io,
            "\$N + D\$ Only (3 lr) & model\\_size($(length(partial_levels.model_size))), \$D\$($(length(partial_levels.data_size))), \$lr\$(3) & $(rs(cost_3lr / pf_day)) & $(rs(rel_partial_3lr)) \\\\",
        )
    end
    println(io, "\$N + D + lr\$ & $partial_lr_level_summary & $(rs(C_partial_lr / pf_day)) & $(rs(rel_partial_lr)) \\\\")
    println(io, "Bayesian Optimization & --- & $(rs(C_bayes / pf_day)) & 1.0 \\\\")
    println(io, "\\hline")
    println(io, "\\end{tabular}")
    println(io, "\\caption{Training cost comparison across experimental design strategies.}")
    println(io, "\\end{table}")
    println(io)
    println(io)

    latex_levels_block(io, "Full Factorial Levels", full_levels)
    latex_levels_block(io, "\$N + D\$ Only Levels", partial_levels)
    latex_levels_block(
        io,
        "\$N + D + lr\$ Levels",
        partial_levels_lr;
        skip=Set([:model_size, :data_size]),
    )

    return String(take!(io))
end

function compute_costs(df::DataFrame)
    C_bayes = compute_bayes_cost(df)
    full_levels, C_full = compute_full_cost(df)
    partial_levels_lr, C_partial_lr = compute_partial_cost(df; include_lr=true)
    partial_levels, C_partial = compute_partial_cost(df; include_lr=false)

    rs(x) = round(x; sigdigits=3)

    println("Full Factorial:")
    println("Cost: $(rs(C_full / pf_day)) pf-day")
    show_levels(full_levels)

    println("\n\nN + D Only:")
    println("Cost: $(rs(C_partial / pf_day)) pf-day")
    println("Cost (3 lr): $(rs(3 * C_partial / pf_day)) pf-day")
    show_levels(partial_levels)

    println("\n\nN + D + lr_prefactor Only:")
    println("Cost: $(rs(C_partial_lr / pf_day)) pf-day")
    show_levels(partial_levels_lr)

    gpu_hours = sum(df.runtime .* df.world_size / (60 * 60))
    println("\n\nBayes Cost: $(rs(C_bayes / pf_day)) pf-day")
    println("Bayes Cost: $(rs(gpu_hours)) GPU-hours")
    println("Rel. Full: $(rs(C_full / C_bayes))")
    println("Rel. Partial (N, D): $(rs(C_partial / C_bayes))")
    println("Rel. Partial (N, D, 3 lr): $(rs(3 * C_partial / C_bayes))")
    println("Rel. Partial (N, D, lr): $(rs(C_partial_lr / C_bayes))")

    latex = generate_cost_latex(
        full_levels,
        C_full,
        partial_levels,
        C_partial,
        partial_levels_lr,
        C_partial_lr,
        C_bayes;
        cost_3lr=3 * C_partial,
    )

    println("\n\nLaTeX:\n")
    println(latex)

    return latex
end

function (@main)(args=[])
    @assert length(args) >= 2 "Usage: brute_force_cost.jl RUNS_JSONL OUTPUT_TEX"

    runs_jsonl = args[1]
    output_tex = args[2]

    @assert isfile(runs_jsonl) "Input JSONL file does not exist: $runs_jsonl"

    df = read_jsonl(runs_jsonl)
    latex = compute_costs(df)

    open(output_tex, "w") do io
        write(io, latex)
    end

    println("\nWrote LaTeX to: $output_tex")

    return 0
end
