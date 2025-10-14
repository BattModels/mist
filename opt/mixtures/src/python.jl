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
    targets = clean_target_name.(pyconvert(Vector{String}, model.config.target_columns))
    rows = map(pyexcess[].evaluate_mixtures(model, mixtures; n, gradients)) do row
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

function label_functional_groups(smi::String)
    pyfg = PythonCall.@pyconst pyimport("electrolyte_fm.interpretibility.functional_groups")
    groups = pyconvert(Vector{String}, pyfg.identify_functional_groups(smi))
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
    rows = map(pyionic[].evaluate_mixtures(model, mixtures; n, fixed_salt)) do row
        row_dict = pyconvert(Dict{String,Any}, row)
        out = Dict{String, Any}(
            "components" => pyconvert(Vector, row_dict["components"]),
            "composition" => pyconvert(Vector, row_dict["composition"]),
            "temperature" => pyconvert(Float64, row_dict["temperature"]),
            "conductivity" => pyconvert(Float64, row_dict["conductivity"]),
            "ln_A" => pyconvert(Float64, row_dict["ln_A"]),
            "Ea" => pyconvert(Float64, row_dict["Ea"]),
            "Tg" => pyconvert(Float64, row_dict["Tg"]),
            "alpha" => pyconvert(Float64, row_dict["alpha"]),
            "beta" => pyconvert(Float64, row_dict["beta"]),
            "lmbda" => pyconvert(Float64, row_dict["lmbda"])
        )
        return out
    end
    return DataFrame(rows)
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
    pred = pyconvert(Dict{String,Any}, pred)
    out = Dict{String, Any}(
        "components" => pyconvert(Vector, pred["components"]),
        "composition" => pyconvert(Vector, pred["composition"]),
        "temperature" => pyconvert(Float64, pred["temperature"]),
        "conductivity" => pyconvert(Float64, pred["conductivity"]),
        "ln_A" => pyconvert(Float64, pred["ln_A"]),
        "Ea" => pyconvert(Float64, pred["Ea"]),
        "Tg" => pyconvert(Float64, pred["Tg"]),
        "alpha" => pyconvert(Float64, pred["alpha"]),
        "beta" => pyconvert(Float64, pred["beta"]),
        "lmbda" => pyconvert(Float64, pred["lmbda"])
    )
    return out
end
