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


geoharmonic_penalty(x, x0::T, penalty::T...) where {T<:Real} = one(T) + harmonic_penalty(x, x0, penalty...)

harmonic_penalty(x, x0::T, penalty::T...) where {T<:Real} = symmetric_polynomial(T(x) - x0, penalty...)

geometric_penalty(x, x0::T, penalty::T...) where {T<:Real} = symmetric_polynomial(log(T(x)) - log(x0), penalty...)

"""
    symmetric_polynomial(x, coefs...)

Evaluate a polynomial of the form: `sum((i, c) -> c * x^(2i), enumerate(coefs))`
"""
function symmetric_polynomial(x, coefs::T...) where {T <: Real}
    y = zero(x)
    for coef in coefs
        x *= x
        y += coef * x
    end
    return y
end
symmetric_polynomial(x, coef::T) where {T <: Real} = coef * x^2

