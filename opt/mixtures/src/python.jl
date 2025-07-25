clean_target_name(x::String) = replace((strip∘first∘split)(x, "["), " " => "_")

load_excess_model(ckpt) = pyexcess[].load_excess_model(ckpt)

function evaluate_mixtures(model::Py, mixtures::Vector{<:Dict}; kwargs...)
    pymixtures = map(mixtures) do mixture
        PythonCall.pydict(;
            compounds=PythonCall.pylist(mixture["compounds"]),
            temperature=mixture["temperature"],
        )
    end
    return evaluate_mixtures(model, pymixtures; kwargs...)
end

function evaluate_mixture(model::Py, compounds::String...; temperature::Real=298.15, kwargs...)
    evaluate_mixtures(model, [Dict("compounds" => compounds, "temperature" => temperature)]; kwargs...)
end

function evaluate_mixtures(model::Py, mixtures::Union{Py, Vector{Py}}; n = 20, gradients=true)
    targets = clean_target_name.(pyconvert(Vector{String}, model.config.target_columns))
    rows = map(pyexcess[].evaluate_mixtures(model, mixtures; n=5, gradients)) do row
        row = pyconvert(Dict{String, Union{Float64, String, Vector}}, row)
        out = Dict(
            "compounds" => row["compounds"],
            "composition" => row["composition"],
            "temperature" => row["temperature"],
        )

        for (idx, target) in enumerate(targets)
            out[target] = row["y"][idx]
            out["$(target)_excess"] = row["y_excess"][idx]
            out["$(target)_linear"] = row["y_linear"][idx]
            if gradients
                out["$(target)_dT"] = row["dy_dT"][idx]
                out["$(target)_dx"] = pyconvert(Vector{Float64}, row["dy_dx"][idx])
                out["$(target)_excess_dT"] = row["dy_excess_dT"][idx]
                out["$(target)_excess_dx"] = pyconvert(Vector{Float64}, row["dy_excess_dx"][idx])
            end
        end

        return out
    end
    return DataFrame(rows)
end

function process_prediction_with_ref(iter::Py, targets::Vector{String}, gradients::Bool)
    return map(iter) do row
        row = pyconvert(Dict{String, Union{Float64, String, Vector}}, row)
        out = Dict(
            "compounds" => row["compounds"],
            "composition" => row["composition"],
            "temperature" => row["temperature"],
        )

        for (idx, target) in enumerate(targets)
            out[target] = row["y"][idx]
            out["$(target)_excess"] = row["y_excess"][idx]
            out["$(target)_linear"] = row["y_linear"][idx]
            out["$(target)_ref"] = row["target_mask"][idx] == 1 ? row["target"][idx] : missing
            out["$(target)_excess_ref"] = row["target_excess_mask"][idx] == 1 ? row["target_excess"][idx] : missing
            if gradients
                out["$(target)_dT"] = row["dy_dT"][idx]
                out["$(target)_dx"] = pyconvert(Vector{Float64}, row["dy_dx"][idx])
                out["$(target)_excess_dT"] = row["dy_excess_dT"][idx]
                out["$(target)_excess_dx"] = pyconvert(Vector{Float64}, row["dy_excess_dx"][idx])
            end
        end
        return out
    end |> DataFrame
end

function evaluate_dataset(model::Py, ds_path; gradients=false)
    targets = clean_target_name.(pyconvert(Vector{String}, model.config.target_columns))
    iter = pyexcess[].evaluate_dataset(model, ds_path; gradients)
    process_prediction_with_ref(iter, targets, gradients)
end

function evaluate_binary_csv(model::Py, ds_path; gradients=false)
    targets = clean_target_name.(pyconvert(Vector{String}, model.config.target_columns))
    iter = pyexcess[].evaluate_binary_csv(model, ds_path; gradients)
    process_prediction_with_ref(iter, targets, gradients)
end
