struct UQReal{T}
    mean::T
    std::T
    n::Int
end

UQReal(x::Vector) = UQReal(mean_and_std(x)..., length(x))
Statistics.mean(x::UQReal) = x.mean
Statistics.std(x::UQReal) = x.std
StatsBase.stderror(x::UQReal) = x.std / sqrt(x.n)
function Base.show(io::IO, x::UQReal)
    μ = mean(x)
    se = stderror(x)
    if get(io, :compact, false)::Bool
        μ = round(μ; sigdigits=5)
        se = round(se; sigdigits=5)
    end
    print(io, "$μ ± $se")
end

Base.:*(x::UQReal, y::Real) = UQReal(x.mean * y, x.std * y, x.n)
Base.:*(x::Real, y::UQReal) = y * x

