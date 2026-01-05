function predict(smi::Vector{String}, model::Py)
    model = model.to("mps")
    model.eval()
    py_out = model.predict(PyList(smi))
    out = Dict{String,Vector{Float64}}()
    for (k, v) in py_out.items()
        out[pyconvert(String, k)] = pyconvert(valtype(out), v["value"])
    end
    df = DataFrame(out)
    insertcols!(df, 1, :smi => smi)
    return df
end

function encode(smi::String; encoding::String="smiles-kekule", random::Bool=false)
    utils = @pyconst(pyimport("electrolyte_fm.data_modules.utils"))
    encoder = utils.MolEncoding(encoding)
    smi_out = random ? encoder.random(smi) : encoder(smi)
    return something(pyconvert(Union{String,Nothing}, smi_out), smi)
end

function embed(smi::Vector{String}, model::Py)
    pyconvert(Matrix, model.embed(PyList(smi)))
end

function predict_monte(smi::Vector{String}, model::Py; n=10)
    model = model.to("mps")
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
    return df
end

encoding_variants(smi::String, n::Int=100) = unique(encode(smi; encoding="smiles", random=true) for _ in 1:n)

function sample_encodings(smi::String, model::Py; n=100)
    smi_variants = encoding_variants(smi, n)
    df = predict(smi_variants, model)
    out = Dict()
    for col in names(df)
        col == "smi" && continue
        out[col] = UQReal(df[:, col])
    end
    return out
end

model_spec(model::Py) = (nothing, model, Colon())
model_spec(name_model::Pair{String,Py}) = (first(name_model), last(name_model), Colon())
model_spec(name_model_cols::Pair{String,<:Pair{Py,<:AbstractVector}}) = (first(name_model_cols), last(name_model_cols)...)
model_spec(model_cols::Pair{Py,<:AbstractVector}) = (nothing, first(model_cols), last(model_cols))

"""
Run all models against a DataFrame of smiles in the `smi_column`

Models can be specified as:

- `model::Py` the model to run
- `"name" => model`: Runs model renaming columns as `col * "_" * name`
- `model => [cols...]`: runs the model only returning the selected columns
- `"name" => model => [cols...]`: Runs the model renaming and selecting columns

By default, a single evaluation is sampled. Increasing `n` will collect multiple samples from the model
"""

function predict_all(df::DataFrame, models...; smi_column=:smi, n=1)
    for spec in models
        name, model, cols = model_spec(spec)
        if n == 1
            df_predict = predict(df[:, smi_column], model)
        else
            df_predict = predict_monte(df[:, smi_column], model; n)
        end
        select!(df_predict, Not(:smi))
        select!(df_predict, cols)
        isnothing(name) || rename!(col -> col * "_" * name, df_predict)
        df = hcat(df, df_predict)
    end
    return df
end

function load_huggingface_model(name::String)
    AutoModel = @pyconst(pyimport("transformers").AutoModel)
    model = AutoModel.from_pretrained(name, trust_remote_code=true)
    return model
end
