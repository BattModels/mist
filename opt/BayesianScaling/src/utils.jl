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

function sample_posterior(f, chains, args::Vector...)
    names = Tuple(chains.name_map[:parameters])
    v = map(names) do name
        p = chains.value[:, name, :]
        reshape(p, 1, size(p)...)
    end
    x = NamedTuple{names}(v)
    return f(args...; x...)
end

logsqdev(x, y) = (log(x) - log(y))^2

"""
    non_embedding_size(d_model, d_ff, n_layers; d_attn=d_model)

The number of non-embedding parameters in a decoder/encoder-only
transformer model, as described in table 1 of

Kaplan, J. et al. 2020. Scaling Laws for Neural Language Models. arXiv.

`d_attn` and `d_model` by default are the same
"""
function non_embedding_size(d_model, d_ff, n_layers; d_attn=d_model)
    attention_qkv = n_layers * 3 * d_model * d_attn
    project = n_layers * d_model * d_attn
    ff = n_layers * 2 * d_model * d_ff
    return attention_qkv + project + ff
end

