module MIST

using PythonCall
using DataFrames
# using Makie
using Statistics: Statistics, mean, std
using StatsBase: StatsBase, stderror, mean_and_std

function finetuned(ckpt)
    MISTFinetuned = pyimport("electrolyte_fm.models.prod_finetune").MISTFinetuned
    model = MISTFinetuned.from_pretrained(ckpt)
    return model
end

function multitask(ckpt)
    MISTMultiTask = pyimport("electrolyte_fm.models.prod_finetune").MISTMultiTask
    return MISTMultiTask.from_pretrained(ckpt)
end

struct UQReal{T}
    mean::T
    std::T
    n::Int
end

Statistics.mean(x::UQReal) = x.mean
Statistics.std(x::Real) = x.std
StatsBase.stderror(x) = x.std / sqrt(x.n)
function Base.show(io::IO, x::UQReal)
    μ = mean(x)
    se = stderror(x)
    if get(io, :compact, false)::Bool
        μ = round(μ; sigdigits=5)
        se = round(se; sigdigits=5)
    end
    print(io, "$μ ± $se")
end

function predict(smi::Vector{String}, model::Py)
    model = model.to("mps")
    py_out = model.predict(PyList(smi))
    out = Dict{String,Vector{Float64}}()
    for (k, v) in py_out.items()
        out[pyconvert(String, k)] = pyconvert(valtype(out), v["value"])
    end
    df = DataFrame(out)
    insertcols!(df, 1, :smi => smi)
    return df
end

function embed(smi::Vector{String}, model::Py)
    pyconvert(Matrix, model.embed(PyList(smi)))
end

function predict_monte(smi::Vector{String}, model::Py; n=10)
    training = model.training
    model = model.train()
    results = Dict()
    py_smi = PyList(smi)
    for _ in 1:n
        sample = model.predict(py_smi)
        for (k, v) in sample.items()
            k = pyconvert(String, k)
            v = pyconvert(Vector{Float64}, v["value"])
            if haskey(results, k)
                results[k] = hcat(results[k], v)
            else
                results[k] = v
            end
        end
    end

    # Compute statistics
    summary = Dict()
    for (k, v) in pairs(results)
        mu_std = map(mean_and_std, eachrow(v))
        summary[k] = map(x -> UQReal(x..., n), mu_std)
    end
    df = DataFrame(summary)
    insertcols!(df, 1, :smi => smi)
    model.train(training)
    return df
end

# @recipe(ErrorLines, x, y, error_y) do scene
#     Attributes()
# end
# Makie.convert_arguments(::Type{<:ErrorLines}, x::Any, y::AbstractVector{<:UQReal}) = (x, mean.(y), stderror.(y))
#
# function Makie.plot!(plt::ErrorLines{<:Tuple{AbstractVector,AbstractVector{<:Real},AbstractVector{<:Real}}})
#     lines!(plt, plt.x, plt.y, Makie.shared_attributes(plt, Lines))
#     if !isnothing(plt.error_y)
#         errorbars!(plt, plt.x, plt.y, plt.error_y, Makie.shared_attributes(plt, Makie.Errorbars))
#     end
#     return plt
# end
#
# @recipe(ErrorCross, x, y, error_x, error_y) do scene
#     Attributes()
# end
#
# Makie.convert_arguments(::Type{<:ErrorCross}, x::AbstractVector{<:UQReal}, y::AbstractVector{<:UQReal}) = (mean.(x), mean.(y), stderror.(x), stderror.(y))
#
# function Makie.plot!(plt::ErrorCross{<:NTuple{4,AbstractVector}})
#     attrs = Makie.shared_attributes(plt, Errorbars)
#     h = errorbars!(plt, plt.x, plt.y, plt.error_y; direction=:y, attrs...)
#     errorbars!(plt, plt.x, plt.y, plt.error_x;
#         color=h.color,
#         colorscale=h.colorscale,
#         colormap=h.colormap,
#         colorrange=h.colorrange,
#         direction=:x,
#         attrs...
#     )
#     return plt
# end

end
