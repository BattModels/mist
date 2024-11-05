function summarize_finetuning(finetuning_export::String)
    data = map(readdir(finetuning_export, join=true)) do file
        data = JSON.parsefile(file; null=missing)
        # Pull out validation loss
        data["val_loss"] = pop!(data, "val_loss_best", pop!(data, "val_loss_last", missing))
        metrics = map(pop!(data, "metrics", [Dict()])) do m
            m["id"] = data["id"]
            m
        end
        return data, metrics
    end

    # Extract Data
    df_runs = DataFrame(first.(data))
    df_metrics = DataFrame(Iterators.flatten(last.(data)))

    return df_runs, df_metrics
end

function join_mertics(df_runs, df_metrics)

    df = select(df, [:id, :dataset, :split, :metric, :lr, :token_group, :best])
    subset!(df, :split => ByRow(==("val")))
    df = combine(groupby(df, [:id, :dataset, :split, :metric, :lr])) do gdf
        gdf = unstack(gdf, :token_group, :best)
        if !(:oov in propertynames(gdf))
            insertcols!(gdf, :lr, :oov => missing; after=true)
        end
        return gdf
    end
    replace!(df.oov, 0 => missing)
    df.oov_split  = @. df.non_oov - df.oov
    return df
end


function widden_crosstab(df)
    df_cross = subset(df,
        :metric => ByRow(x -> startswith(x, "crosstab")),
        :bootstrap => ByRow(ismissing),
        :type => ByRow(==("last")),
        :tok_group => ByRow(in(["oov", "non_oov", "all"]))
    )
    select!(df_cross, Not([:bootstrap, :type]))

    # Populate table
    df_cross = combine(groupby(df_cross, [:id, :split, :tok_group])) do gdf
        xtab = unstack(gdf, :metric, :value, renamecols=x -> Symbol(split(x, "_")[2]))
        select!(xtab, sort(propertynames(xtab)))

        xtab.nobs .= Int.(xtab.tp + xtab.fp + xtab.tn + xtab.fn)
        xtab.tpr .= xtab.tp / (xtab.tp + xtab.fn)
        xtab.tnr .= xtab.tn / (xtab.tn + xtab.fp)
        xtab.ppv .= xtab.tp / (xtab.tp + xtab.fp)
        xtab.npv .= xtab.tn / (xtab.tn + xtab.fn)
        xtab
    end
    return df_cross
end
