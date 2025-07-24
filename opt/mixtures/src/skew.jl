function argextreme(x, y)
    x = x[(!ismissing).(y)]
    indices = eachindex(y)[(!ismissing).(y)]
    y = y[(!ismissing).(y)]
    if length(x) == 0
        return missing, missing, missing
    end
    if abs(maximum(y)) > abs(minimum(y))
        idx = argmax(y)
    else
        idx = argmin(y)
    end
    return indices[idx], x[idx], y[idx]
end

function excess_skew(df; targets=["density", "molar_volume", "molar_enthalpy"])
    df = transform(df, :compounds => ByRow(sort) => :compound_id)
    skew = combine(groupby(df, :compound_id)) do gdf
        out = Dict()
        for target in targets
            # Model
            x1 = first.(gdf.composition)
            y = gdf[!, "$(target)_excess"]
            idx, xe, ye = argextreme(x1, y)
            out[target] = abs(0.5 - xe)
            out["$(target)_value"] = ye
            out["$(target)_rel"] = ye / (ismissing(idx) ? missing : gdf[idx, target])

            # Reference
            y = gdf[!, "$(target)_excess_ref"]
            idx, xe, ye = argextreme(x1, y)
            out["$(target)_ref"] = abs(0.5 - xe)
            out["$(target)_value_ref"] = ye
            out["$(target)_rel_ref"] = ye / (ismissing(idx) ? missing : gdf[idx, "$(target)_ref"])
        end
        return NamedTuple(Symbol(k) => v for (k, v) in pairs(out))
    end
end
