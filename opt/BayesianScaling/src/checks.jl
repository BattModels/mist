"""
Compute correlations between the model's residuals and other possibly explanatory variables
"""
function residual_correlations(error::Vector, df)
    # Compute correlations between the residuals
    cols = names(df, eltype.(eachcol(df)) .<: Real)
    rescor = Dict{eltype(cols),Any}()
    for col in cols
        rescor[col] = StatsBase.corspearman(error, float.(df[!, col]))
    end
    return rescor
end

