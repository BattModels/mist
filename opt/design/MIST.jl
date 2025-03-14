module MIST

using PythonCall
using DataFrames
using Statistics: mean, std

function finetuned(ckpt)
    MISTFinetuned = @pyconst(pyimport("electrolyte_fm.models.prod_finetune")).MISTFinetuned
    return MISTFinetuned.from_pretrained(ckpt)
end

function multitask(ckpt)
    MISTMultiTask = @pyconst(pyimport("electrolyte_fm.models.prod_finetune")).MISTMultiTask
    return MISTMultiTask.from_pretrained(ckpt)
end

function predict(smi::Vector{String}, model::Py)
    py_out = model.predict(PyList(smi))
    out = []
    for py_y in py_out
        y = Dict{String,Float64}()
        for (k, v) in py_y.items()
            y[pyconvert(String, k)] = pyconvert(Float64, v["value"])
        end
        push!(out, y)
    end
    df = DataFrame(out)
    insertcols!(df, 1, :smi => smi)
    return df
end

function embed(smi::Vector{String}, model::Py)
    pyconvert(Matrix, model.embed(PyList(smi)))
end

function predict_monte(smi::Vector{String}, model::Py; n=10)
    model = model.train()
    results = Dict()
    for i in 1:n
        df_sample = predict(smi, model)
        sample = Dict(zip(names(df_sample), eachcol(df_sample)))
        pop!(sample, "smi")
        for (k, v) in sample
            if haskey(results, k)
                results[k] = hcat(results[k], vec(v))
            else
                results[k] = v
            end
        end
    end

    # Compute statistics
    summary = Dict()
    for (k, v) in pairs(results)
        summary["$(k)_mean"] = vec(mean(v; dims=2))
        summary["$(k)_stderr"] = vec(std(v; dims=2)) / sqrt(n)
    end
    df = DataFrame(summary)
    insertcols!(df, 1, :smi => smi)
    return df
end

end
