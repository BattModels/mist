function logrange(lb, ub; base=10, kwargs...)
    iter = range(log(base, lb), log(base, ub); kwargs...)
    if base == 10
        f = exp10
    elseif base == 2
        f = exp2
    else
        f = x -> base^x
    end
    return Iterators.map(f, iter)
end

function sample_posterior(f, chains, args::Union{Real,Vector}...)
    x = get(chains, chains.name_map[:parameters])
    x = NamedTuple{keys(x)}(map(vec, values(x)))
    args = map(transpose, args)
    return f(args...; x...) |> Array
end

"""
Condition a model on a sampled chain
"""
function Turing.condition(model, sample)
    x = get(sample, sample.name_map[:parameters])
    x = NamedTuple{keys(x)}(map(only, values(x)))
    return Turing.condition(model, x)
end

logsqdev(x, y) = (log(x) - log(y))^2
