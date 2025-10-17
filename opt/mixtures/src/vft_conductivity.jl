const SOLVENT_DATA = Dict(

    # Source: https://landtinst.com/
    "PC"  => (density_g_cm3 = 1.2000, mp_C = -48.8, MW_g_mol = 102),
    "EMC" => (density_g_cm3 = 1.0071, mp_C = -53, MW_g_mol = 104),
    "DMC" => (density_g_cm3 = 1.06311, mp_C = 4, MW_g_mol = 90),
    "DEC" => (density_g_cm3 = 0.9730, mp_C = -43, MW_g_mol = 118),
    "EC"  => (density_g_cm3 = 1.3210, mp_C = 36.4, MW_g_mol = 88),
    # Density/ MP: https://www.chemicalbook.com/ChemicalProductProperty_EN_CB9420252.htm
    # MW: https://www.sigmaaldrich.com/US/en/product/aldrich/901686
    "FEC" => (density_g_cm3 = 1.45, mp_C = 20.5, MW_g_mol = 106.05),
)

function one_molar_to_mole_fraction(solvent)
    # 1 M = 1 mole / L
   data = SOLVENT_DATA[solvent]
   mass_g = data.density_g_cm3 * 1000  # mass of 1 L solvent in grams
   n_solvent = mass_g / data.MW_g_mol  # moles of solvent
   n_solute = 1.0  # 1 M solution = 1 mole solute per L
   return n_solute/(n_solvent + n_solute)
end

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
