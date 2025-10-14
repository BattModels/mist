function calculate_excess!(df::DataFrame)
    comp_matrix = reduce(hcat, df.composition)'

    # Update salt composition to account for discretization
    salt_comp = first(comp_matrix[:, 4])
    solvent_comps = comp_matrix[:, 1:3]
    @assert all(count(!iszero, solvent_comps[i, :]) == 1 for i in 1:3) "First three rows must be single solvent mixtures"

    pure_solvent_param = Dict(
        "Ea" => df[1:3, "Ea"],
        "Tg" => df[1:3, "Tg"],
    )

    solvent_fractions = solvent_comps ./ sum(solvent_comps, dims = 2)
    for parameter in ["Ea", "Tg"]
        pure_solvent_values = pure_solvent_param[parameter]
        df[!, "ideal_mixing_$(parameter)"] = solvent_fractions * pure_solvent_values
        df[!, "excess_$(parameter)"] = df[!, parameter] .- df[!, "ideal_mixing_$(parameter)"]
        df[!, "relative_excess_$(parameter)"] = abs.(df[!, "excess_$(parameter)"] ./ df[!, parameter])
        @assert sum(abs.(collect(df[1:3, "excess_$(parameter)"]))) < 1e-6 "Zero excess for expected degenerate case"
    end
    return df
end


function conductivity_composition_curve(model, mixture; n = 50)
    x1 = 1.0
    x2 = 0.0
    x3 = 0.0
    composition = range(0.02, stop = 0.20, length = n)
    conductivity = [
        evaluate_at_composition(model, mixture, [x1 - x, x2, x3, x])["conductivity"] for x in
                                                                                                  composition
    ]
    return composition, conductivity
end
