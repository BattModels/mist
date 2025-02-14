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
        push!(exprs, :(logpdf_prior(priors.$name, θ.$name)))
    end
    return Expr(:call, +, exprs...)
end

function logpdf_prior(priors::Tuple{Vararg{<:Distribution}}, θ)
    return sum(zip(priors, θ)) do (p, x)
        logpdf_prior(p, x)
    end
end

logpdf_prior(prior::Distribution, θ) = logpdf(prior, θ)

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
sample_priors(m::Tuple{Vararg{<:Distribution}}) = map(sample_priors, m)
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
transform_support(p::Tuple{Vararg{<:Distribution}}) = as(map(transform_support, p))
TransformVariables.as(t::NamedTuple) = as(map(TransformVariables.as, t))

function init_logdensity_model(m, adtype=:Enzyme)
    t = transform_support(m)
    p = TransformedLogDensity(t, m)
    ∇P = ADgradient(adtype, p)
    return ∇P
end

"""
    y = transform_samples(t, x::AbstractMatrix{T, 3})

Apply the transform `t` to each column of `x` and return a `ComponentArray` of the transformed samples.
"""
function transform_samples(t::TransformVariables.AbstractTransform, x::AbstractArray{T,3}) where {T<:Real}
    y = similar(x)
    ax = ComponentArrays.Axis(t)
    y = ComponentArray(y, FlatAxis(), FlatAxis(), ax)
    for (xs, ys) in zip(eachslice(x; dims=(1, 2)), eachslice(y; dims=(1, 2)))
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
            if t isa TransformVariables.ScalarTransform
                y[index] = TransformVariables.transform(t, x[index])
            else
                y[index] = TransformVariables.transform(t, x[index:index]) |> first
            end
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
function ComponentArrays.Axis(tt::TransformVariables.TransformTuple{<:NamedTuple})
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

function sample_chains(ℓ; nchains=15, draws=1_000, adtype=:Enzyme)
    model = init_logdensity_model(ℓ, adtype)
    d = dimension(model)
    reporter = DynamicHMC.NoProgressReport()
    posterior = Vector{Matrix{Float64}}(undef, nchains)
    Threads.@threads :dynamic for i in ProgressBar(1:nchains)
        posterior[i] = mcmc_with_warmup(Random.default_rng(), model, draws; reporter).posterior_matrix
    end
    y_raw = stack(posterior'; dims=2)
    @assert size(y_raw) == (draws, nchains, d)
    y = transform_samples(model.ℓ.transformation, y_raw)

    # Add ComponentVectors to the chains
    ax = ComponentArrays.Axis(model.ℓ.transformation)
    y = ComponentArray(y, FlatAxis(), FlatAxis(), ax)
    y_raw = ComponentArray(y_raw, FlatAxis(), FlatAxis(), ax)
    return y, y_raw
end

"""
    save_results(model, chains, raw_chains; outdir)

Save the model and sample chains to the output directory
"""
function save_results(model, chains, raw_chains; outdir=joinpath(pkgdir(@__MODULE__), "out"))
    model_name = string(uuid4())
    outdir = joinpath(outdir, model_name)
    mkpath(outdir)
    metadata = (;
        git=readchomp(`git describe --all --long --dirty`),
        timestamp=string(Dates.now()),
    )
    jldsave(joinpath(outdir, "chains.jld2"); model, chains, raw_chains, metadata)
    return outdir
end

mapchains(f, op, chains::AbstractArray{<:Real,3}) = mapreduce(f, op, eachslice(chains, dims=(1, 2)))

function mapchains(f, chains::AbstractArray{<:Real,3}, x::Vector)
    out = similar(chains, size(chains, 1), size(chains, 2), length(x))
    for I in CartesianIndices(axes(chains)[1:2])
        for (rdx, x) in enumerate(x)
        θ = @view chains[I.I..., :]
            out[I, rdx] = f(x, θ)
        end
    end
    return out
end

"""
    chain = subsample(chains::AbstractArray{T,3}, n::Integer)

Draw `n` samples from each chain in `chains` of size `(draws, chains, parameters)`
"""
function subsample(chains::AbstractArray{T,3}, n::Integer) where {T}
    s = similar(chains, n, 1, size(chains, 3))
    draws = CartesianIndices((axes(chains, 1), axes(chains, 2)))
    for cdx in 1:n
        i = rand(draws)
        θ = @view chains[i.I..., :]
        copyto!(selectdim(s, 1, cdx), θ)
    end
    if chains isa ComponentArray
        s = ComponentArray(s, chains.axes...)
    end
    return s
end

function subsample(chains::AbstractArray{T,3}, p::AbstractFloat=0.2) where {T}
    @assert 0 < p <= 1 lazy"Expected p ∈ (0, 1], got $p"
    ns = size(chains, 1) * size(chains, 2)
    n = ceil(Int, ns * p)
    return subsample(chains, n)
end


"""
    μ, l, u = credible_interval(chains; p=0.95)

Return the expectation of `chains` and the `p`-percentile credible interval
"""
function credible_interval(chains::AbstractArray{T,3}; p=0.95) where {T}
    p = (1 - p) / 2
    μ = vec(mean(chains; dims=(1, 2)))
    l = similar(μ)
    u = similar(l)
    for (idx, chains) in enumerate(eachslice(chains; dims=3))
        l[idx], u[idx] = quantile(chains, (p, 1 - p))
    end
    return μ, l, u
end
function credible_interval(chains::AbstractMatrix{T}; p::AbstractFloat=0.95) where {T}
    p = (1 - p) / 2
    return mean(chains), quantile(vec(chains), (p, 1 - p))...
end


"""
Compute `y/exp(E[log(ŷ) - log(P)])` where `y` is the response, `ŷ` is the expected response and `P` is a penalty term
It's expected that `ŷ = f(x...) × P`
"""
function penalty_residual(y::Vector{T}, y_hat::AbstractArray{T,3}, penalty::AbstractArray{T,3}) where {T<:Real}
    mu = similar(y_hat)
    for I in eachindex(IndexCartesian(), y_hat)
        mu[I] = log(y_hat[I]) - penalty[I]
    end
    mu = vec(mean(mu; dims=(1, 2)))
    return xexpy.(y, -mu)
end

