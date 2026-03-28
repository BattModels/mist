#!/usr/bin/env -S julia +release --color=auto --startup-file=no --project=@script
# Estimate the compute cost of doing a brute force search
# Usage: ./brute_force_cost.jl ../out/dec_3_sweep_runs.jsonl
using DataFrames
using JSON: JSON
using BayesianScaling: non_embedding_size

const AVG_SEQ_LENGTH::Float64 = 65.2
"""
Average number of tokens per molecule for the Smirk Tokenizer on REALSpace
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
    # LR Prefector
    # Round to avoid 3.125 and 3.1249999... being treated differently
    lr_prefactor = @. df.lr / sqrt(df.effective_batch_size)
    return unique(round.(lr_prefactor; sigdigits=6))
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

function compute_partial_cost(df::DataFrame; include_lr::Bool = true)
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
        6 * p.model_size * p.data_size * AVG_SEQ_LENGTH
    end
    return levels, total_cost
end

function compute_bayes_cost(df::DataFrame)
    N = df.model_size
    D = @. df.effective_batch_size * df.max_steps * AVG_SEQ_LENGTH
    return sum(x -> 6*prod(x), zip(N, D))
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

function compute_costs(df)
    C_bayes = compute_bayes_cost(df)
    full_levels, C_full = compute_full_cost(df)
    partial_levels_lr, C_partial_lr = compute_partial_cost(df; include_lr = true)
    partial_levels, C_partial = compute_partial_cost(df; include_lr = false)

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

    # Cost Savings
    gpu_hours = sum(df.runtime .* df.world_size / (60 * 60))
    println("\n\nBayes Cost: $(rs(C_bayes / pf_day)) pf-day")
    println("Bayes Cost: $(rs(gpu_hours)) GPU-hours")
    println("Rel. Full: $(rs(C_full / C_bayes))")
    println("Rel. Partial (N, D): $(rs(C_partial / C_bayes))")
    println("Rel. Partial (N, D, 3 lr): $(rs(3 * C_partial / C_bayes))")
    println("Rel. Partial (N, D, lr): $(rs(C_partial_lr / C_bayes))")

    return nothing
end

function (@main)(args=[])
    runs_jsonl = args[1]
    @assert isfile(runs_jsonl)
    df = read_jsonl(runs_jsonl)
    compute_costs(df)
    return 0
end
