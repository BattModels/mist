"""
Compute correlations between the model's residuals and other possibly explanatory variables
"""
function residual_correlations(model, chains, df)
    y = first(model.args)
    y_hat = sample_response(model, chains)
    return residual_correlations(y .- y_hat, df)
end

function residual_correlations(error, df)
    # Compute correlations between the residuals
    cols = names(df, eltype.(eachcol(df)) .<: Real)
    push!(cols, "min_val_loss")
    rescor = Dict{eltype(cols),Any}()
    for col in cols
        cor = StatsBase.corspearman(error, df[!, col])
        rescor[col] = (; μ=mean(cor), σ=std(cor))
    end
    @info rescor
    return rescor
end
