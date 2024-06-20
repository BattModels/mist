function logrange(lb, ub; base=10, kwargs...)
    iter = range(log(base, lb), log(base, ub); kwargs...)
    if base == 10
        f = exp10
    elseif base == 2
        f = exp2
    else
        f = x -> base^x
    end
    return map(f, iter)
end

"""
    sample_posterior(f, chains, args...)

Applies a function `f(args...; θ...)` using samples `θ` from `chains`
"""
function sample_posterior(f, chains, args::Real...)
    names = Tuple(chains.name_map[:parameters])
    x = map(names) do name
        name => chains[:, name, :]
    end |> NamedTuple
    return f(args...; x...)
end

extract_param(chain, name) = chain[:, name, :]

function sample_posterior(f, chains, args::Vector...)
    names = Tuple(chains.name_map[:parameters])
    v = map(names) do name
        p = chains.value[:, name, :]
        reshape(p, 1, size(p)...)
    end
    x = NamedTuple{names}(v)
    return f(args...; x...)
end

"""
Condition a model on a sampled chain
"""
function Turing.condition(model, sample)
    names = Tuple(sample.name_map[:parameters])
    x = NamedTuple{names}(map(p -> only(sample.value[:, p, :]), names))
    return Turing.condition(model, x)
end

logsqdev(x, y) = (log(x) - log(y))^2

"""
The number of non-embedding parameters in a decoder/encoder-only
transformer model, as described in table 1 of

Kaplan, J. et al. 2020. Scaling Laws for Neural Language Models. arXiv.

`d_attn` and `d_model` by default are the same
"""
function non_embedding_size(d_model, d_ff, n_layers; d_attn=d_model)
    attention_qkv = n_layers * d_model * 2 * d_attn
    project = n_layers * d_model * d_attn
    ff = n_layers = 2 * d_model * d_ff
    return attention_qkv + project + ff
end

function sample_response(model, chains)
    y = stack(generated_quantities(model, chains))
    σ² = chains[:, :σ², :]
    for idx in CartesianIndices(size(y)[2:end])
        for i in 1:size(y, 1)
            y[i, idx] = rand(LogNormal(log(y[i, idx]), σ²[idx]))
        end
    end
    y = reshape(y, size(y, 1), :)
    return y
end

""" Compute quantiles for each column of `x` at `p` """
function col_quantile(x::AbstractMatrix, p)
    q = Matrix{Float32}(undef, length(p), size(x, 2))
    for (i, col) in enumerate(eachcol(x))
        q[:, i] .= quantile(col, p)
    end
    return q
end
