struct BayesModel{P,X}
    priors::P
    covariates::X
end

"""
    logpdf_prior(priors::NamedTuple, θ)

Compute the logdensity of the priors `priors` at `θ`, expects `θ` to define
`getproperty` for each key in `priors` (i.e. `θ` is a `NamedTuple` or `ComponentVector`).
"""
@generated function logpdf_prior(priors::NamedTuple{names}, θ) where {names}
    exprs = []
    for name in names
        push!(exprs, :(logpdf(priors.$name, θ.$name)))
    end
    return Expr(:call, +, exprs...)
end

"""
    logpdf_prior_vec(priors::NamedTuple, θ::AbstractVector)

Compute the logdensity of the priors `priors` at `θ`, expects a 1:1 mapping of elements in `θ` to `priors`.
(i.e. each distribution in `priors` is univariate)
"""
@generated function logpdf_prior_vec(priors::NamedTuple{names}, θ::AbstractVector) where {names}
    exprs = []
    for (idx, name) in enumerate(names)
        push!(exprs, :(logpdf(priors.$name, θ[$idx])))
    end
    return Expr(:call, +, exprs...)
end

sample_priors(m::BayesModel) = sample_priors(m.priors)
sample_priors(m::NamedTuple) = NamedTuple{keys(m)}(map(sample_priors, values(m)))
sample_priors(m::Distribution) = rand(m)

function evaluate_model_prior(model, n=100)
    ℓ = 0
    for _ in 1:n
        θ = ComponentVector(; sample_priors(model)...)
        ℓ += model(θ)
    end
    return ℓ / n
end

"""
    t = transform_support(prior)

Construct a transform using [TransfromVariables](https://github.com/tpapp/TransformVariables.jl) for mapping ℝ to the support of `prior`
`prior` can be a `NamedTuple` of distributions or a single distribution.
"""
function transform_support(d::UnivariateDistribution)
    (; lb, ub) = support(d)
    if lb == -Inf && ub == Inf
        return as_real
    elseif lb == -Inf
        return as_negative_real
    elseif ub == Inf
        return as_positive_real
    end
    return as(Real, lb, ub)
end
transform_support(p::NamedTuple) = as(map(transform_support, p))
TransformVariables.as(t::NamedTuple) = as(map(TransformVariables.as, t))

function init_logdensity_model(m, adtype=:Enzyme)
    t = transform_support(m)
    p = TransformedLogDensity(t, m)
    ∇P = ADgradient(adtype, p)
    return ∇P
end

"""
    y = transform_samples(t, x::AbstractMatrix)

Apply the transform `t` to each column of `x` and return a `ComponentArray` of the transformed samples.
"""
function transform_samples(t::TransformVariables.AbstractTransform, x::AbstractMatrix)
    y = similar(x)
    for idx in axes(x, 2)
        xs = selectdim(x, 2, idx)
        ys = selectdim(y, 2, idx)
        transform!(ys, t, xs)
    end
    return y
end

function transform!(y::AbstractVector, tt::TransformVariables.TransformTuple, x::AbstractVector)
    (; transformations) = tt
    @assert TransformVariables.dimension(tt) == length(y) == length(x)
    index = firstindex(y)
    for t in transformations
        d = TransformVariables.dimension(t)
        if d == 1
            y[index] = TransformVariables.transform(t, x[index])
        else
            si = range(index; length=d)
            ys = view(y, si)
            xs = view(x, si)
            transform!(ys, t, xs)
        end
        index += d
    end
    return y
end
transform!(y::AbstractVector, t::TransformVariables.AbstractTransform, x::AbstractVector) = copyto!(y, TransformVariables.transform(t, x))

# Construct a ComponentArray Axis for a TransformVariables's transform
function ComponentArrays.Axis(tt::TransformVariables.TransformTuple)
    ax_tt = []
    index = 1
    for (k, t) in pairs(tt.transformations)
        ax = ComponentArrays.Axis(t)
        n = TransformVariables.dimension(t)
        if ax isa Union{ComponentArrays.ShapedAxis,ComponentArrays.Axis}
            ax = ComponentArrays.ViewAxis(range(index; length=n), ax)
        else
            ax = ComponentArrays.reindex(ax, index - 1)
        end
        index += TransformVariables.dimension(t)
        push!(ax_tt, k => ax)
    end
    return ComponentArrays.Axis(NamedTuple(ax_tt))
end

function ComponentArrays.Axis(t::TransformVariables.AbstractTransform)
    if TransformVariables.dimension(t) == 1
        return ComponentArrays.ViewAxis(1)
    end
    return ComponentArrays.ViewAxis(range(1; length=TransformVariables.dimension(t)))
end

function ComponentArrays.Axis(t::TransformVariables.ArrayTransformation)
    if length(t.dims) == 1
        return ComponentArrays.ViewAxis(1:t.dims[1])
    else
        return ComponentArrays.ShapedAxis(t.dims)
    end
end

function sample_chains(model; nchains=15, draws=1_000)
    d = dimension(model)
    nchains *= ceil(Int, sqrt(d))
    raw_samples = Array{Float64}(undef, d, draws, nchains)
    for i in 1:nchains
        result = DynamicHMC.mcmc_with_warmup(Random.default_rng(), model, draws; reporter=DynamicHMC.ProgressMeterReport())
        raw_samples[:, :, i] .= result.posterior_matrix
    end
    rs = reshape(raw_samples, :, draws * nchains)
    @assert size(rs, 1) == d
    y = transform_samples(model.ℓ.transformation, rs)

    # Permute to (draws, nchains, d)
    y = permutedims(reshape(y, d, draws, nchains), (2, 3, 1))
    yr = permutedims(raw_samples, (2, 3, 1))

    # Add ComponentVectors to the chains
    ax = ComponentArrays.Axis(model.ℓ.transformation)
    y = ComponentArray(y, FlatAxis(), FlatAxis(), ax)
    yr = ComponentArray(yr, FlatAxis(), FlatAxis(), ax)
    return y, yr
end
