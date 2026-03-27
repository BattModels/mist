#!/usr/bin/env -S julia +release --color=auto --startup-file=no --project=@script
# Estimate the compute cost of doing a brute force search
# Usage: ./brute_force_cost.jl ../out/dec_3_sweep_runs.jsonl
using DataFrames
using JSON: JSON
using BayesianScaling: non_embedding_size

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
    return round.(lr_prefactor; sigdigits=6)
end

function compute_full_cost(df::DataFrame)
    levels = (;
        d_model = unique(df.d_model),
        n_layer = unique(df.d_model ./ df.aspect_ratio),
        ff_ratio = unique(df.d_model),
        effective_batch_size = unique(df.effective_batch_size),
        kv_size = unique(df.kv_size),
        data_size = unique(df.max_steps .* df.effective_batch_size),
        lr_prefactor = unique_lr_prefactor(df),
    )
    l_names = keys(levels)
    return sum(Iterators.product(levels...)) do p
        p = NamedTuple{l_names}(p)
        N = non_embedding_size(p.d_model, p.ff_ratio, p.n_layer)
        D = p.data_size
        C = 6 * N * D
    end
end

function compute_partial_cost(df::DataFrame)
    levels = (;
        model_size = unique(df.model_size),
        data_size = unique(df.effective_batch_size .* df.max_steps),
        # lr_prefactor = unique_lr_prefactor(df),
    )
    l_names = keys(levels)
    return sum(Iterators.product(levels...)) do p
        p = NamedTuple{l_names}(p)
        C = 6 * p.model_size * p.data_size
    end
end

function compute_bayes_cost(df::DataFrame)
    N = df.model_size
    D = @. df.effective_batch_size * df.max_steps
    return sum(x -> 6*prod(x), zip(N, D))
end

function estimate_cost(levels::NamedTuple)
    l_names = keys(levels)
    total_compute = sum(Iterators.product(levels...)) do p
        p = NamedTuple{l_names}(p)
        N = non_embedding_size(p.d_model, p.ff_ratio, p.n_layer)
        D = p.data_size
        C = 6 * N * D
    end
    return total_compute
end

function (@main)(args=[])
    runs_jsonl = args[1]
    @assert isfile(runs_jsonl)
    df = read_jsonl(runs_jsonl)

    @show C_bayes = compute_bayes_cost(df)
    @show C_full = compute_full_cost(df)
    @show C_partial = compute_partial_cost(df)

    # Cost Savings
    @show C_full / C_bayes
    @show C_partial / C_bayes


    return 0
end
