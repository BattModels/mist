struct ElementwiseVariance{T,S,W} <: OnlineStat{T}
    σ2::S
    μ::T
    weight::W
    n::Ref{Int}
end

function ElementwiseVariance(T::Type{<:Number}, n::Integer; weight=EqualWeight())
    s = Vector{T}(undef, n)
    m = Vector{T}(undef, n)
    s .= zero(T)^2 / one(T)
    m .= zero(T) / one(T)
    ElementwiseVariance{Vector{T},typeof(s),typeof(weight)}(s, m, weight, 0)
end

OnlineStatsBase.fit!(o::ElementwiseVariance, x::AbstractVector{T}) where {T} = (OnlineStatsBase._fit!(o, x); return o)
function OnlineStatsBase._fit!(o::ElementwiseVariance, x)
    μ = o.μ
    γ = o.weight(o.n[] += 1)
    @. o.μ = smooth(o.μ, x, γ)
    @. o.σ2 = smooth(o.σ2, (x - o.μ) * (x - μ), γ)
    return o
end

function OnlineStatsBase._merge!(o::ElementwiseVariance, o2::ElementwiseVariance)
    γ = o2.n / (o.n += o2.n)
    @. δ = o2.μ - o.μ
    @. o.σ2 = smooth(o.σ2, o2.σ2, γ) + δ^2 * γ * (1.0 - γ)
    @. o.μ = smooth(o.μ, o2.μ, γ)
    return o
end

function OnlineStatsBase.value(o::ElementwiseVariance)
    if nobs(o) > 0
        return @. o.σ2 * bessel(o)
    else
        return NaN
    end
end

StatsBase.var(o::ElementwiseVariance) = OnlineStatsBase.value(o)
StatsBase.mean(o::ElementwiseVariance) = o.μ
StatsBase.nobs(o::ElementwiseVariance) = o.n[]


