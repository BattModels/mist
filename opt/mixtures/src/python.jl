clean_target_name(x::String) = replace((strip∘first∘split)(x, "["), " " => "_")

load_excess_model(ckpt) = pyexcess[].load_excess_model(ckpt)
load_conductivity_model(ckpt) = pyionic[].load_conductivity_model(ckpt)

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
    targets = excess_target_names(model)
    rows = [excess_row(row, targets; gradients)
            for row in pyexcess[].evaluate_mixtures(model, mixtures; n, gradients)]
    return DataFrame(rows)
end

excess_target_names(model::Py) = [clean_target_name(pyconvert(String, name)) for name in model.config.target_columns]
float_value(value::Py) = pyconvert(Float64, PythonCall.pyfloat(value))

function excess_row(row::Py, targets; gradients=false, references=false)
    out = Dict{String,Any}(
        "compounds" => [pyconvert(String, value) for value in row["compounds"]],
        "composition" => [float_value(value) for value in row["composition"]],
        "temperature" => float_value(row["temperature"]),
    )
    for (index, target) in enumerate(targets)
        idx = index - 1 # The remaining values are still Python objects.
        out[target] = float_value(row["y"][idx])
        out["$(target)_excess"] = float_value(row["y_excess"][idx])
        out["$(target)_linear"] = float_value(row["y_linear"][idx])
        if references
            out["$(target)_ref"] = float_value(row["target_mask"][idx]) == 1 ? float_value(row["target"][idx]) : missing
            out["$(target)_excess_ref"] = float_value(row["target_excess_mask"][idx]) == 1 ? float_value(row["target_excess"][idx]) : missing
        end
        if gradients
            out["$(target)_dT"] = float_value(row["dy_dT"][idx])
            out["$(target)_dx"] = [float_value(value) for value in row["dy_dx"][idx]]
            out["$(target)_excess_dT"] = float_value(row["dy_excess_dT"][idx])
            out["$(target)_excess_dx"] = [float_value(value) for value in row["dy_excess_dx"][idx]]
        end
    end
    return out
end

function process_prediction_with_ref(iter::Py, targets::Vector{String}, gradients::Bool)
    DataFrame([excess_row(row, targets; gradients, references=true) for row in iter])
end

function evaluate_dataset(model::Py, ds_path; gradients=false)
    targets = excess_target_names(model)
    iter = pyexcess[].evaluate_dataset(model, ds_path; gradients)
    process_prediction_with_ref(iter, targets, gradients)
end

function evaluate_binary_csv(model::Py, ds_path; gradients=false)
    targets = excess_target_names(model)
    iter = pyexcess[].evaluate_binary_csv(model, ds_path; gradients)
    process_prediction_with_ref(iter, targets, gradients)
end

function label_functional_groups(smi::String)
    pyfg = PythonCall.@pyconst pyimport("electrolyte_fm.interpretibility.functional_groups")
    groups = [pyconvert(String, value) for value in pyfg.identify_functional_groups(smi)]
    length(groups) == 0 ? ["Other"] : groups
end


function evaluate_conductivity(
    model::Py,
    mixtures::Vector{<:Dict};
    n::Integer=20,
    fixed_salt::Union{Nothing,Real}=nothing,
    kwargs...)

    pymixtures = map(mixtures) do mixture
        PythonCall.pydict(;
        solvents    = PythonCall.pylist(mixture["solvents"]),
        salt        = PythonCall.pylist(mixture["salt"]),
        temperature = mixture["temperature"],
        )
    end
    return evaluate_conductivity(model, pymixtures; n=n, fixed_salt=fixed_salt, kwargs...)
end

function evaluate_conductivity(
    model::Py,
    mixtures::Union{Py, Vector{Py}};
    n::Integer=20,
    fixed_salt::Union{Nothing,Real}=nothing,
    kwargs...)
    rows = map(conductivity_row, pyionic[].evaluate_mixtures(model, mixtures; n, fixed_salt))
    return DataFrame(rows)
end

function conductivity_row(row::Py)
    # Convert the Python values once. Avoid generic container conversion here:
    # it can already convert values to Julia before the scalar pyconvert call.
    out = Dict{String, Any}(
        "components" => [pyconvert(String, value) for value in row["components"]],
        "composition" => [pyconvert(Float64, value) for value in row["composition"]],
    )
    for name in ["temperature", "conductivity", "ln_A", "Ea", "Tg", "alpha", "beta", "lmbda"]
        out[name] = pyconvert(Float64, row[name])
    end
    return out
end

function evaluate_at_composition(model::Py, mixture::Dict, composition::Vector{Float64})
    pymixture = PythonCall.pydict(;
        solvents=PythonCall.pylist(mixture["solvents"]),
        salt=PythonCall.pylist(mixture["salt"]),
        temperature=mixture["temperature"],
    )
    pycomposition = PythonCall.pylist(composition)
    return evaluate_at_composition(model, pymixture, pycomposition)
end

function evaluate_at_composition(model::Py, mixture::Py, composition::Py)
    pymixture = PythonCall.pydict(;
        solvents=PythonCall.pylist(mixture["solvents"]),
        salt=PythonCall.pylist(mixture["salt"]),
        temperature=mixture["temperature"],
    )
    pycomposition = PythonCall.pylist(composition)
    pred = pyionic[].evaluate_at_composition(model, pymixture, pycomposition)
    return conductivity_row(pred)
end
