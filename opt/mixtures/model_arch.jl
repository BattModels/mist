using GLM
using DataFrames
using CSV: CSV
using CategoricalArrays: categorical


function (@main)(ARGS=[])
    df_runs = DataFrame(CSV.File(joinpath(@__DIR__, "excess_model_sweep_v1.csv")))
    rename!(df_runs,
        "val/loss" => "loss",
        "model.relative_excess" => "relative_excess",
        "model.num_control" => "num_control",
        "model.interactions" => "interactions",
        "model.temperature_dependence" => "temperature_dependence",
        "trainer.freeze.thaw_depth" => "thaw_depth",
        "trainer.lr" => "lr",
    )
    df_runs.interactions = categorical(df_runs.interactions)
    model = lm(
        @formula(loss ~ num_control + temperature_dependence + interactions + thaw_depth + log(lr) + log(lr)^2),
        df_runs;
        contrasts = Dict(
            :interactions => EffectsCoding(; base="difference"),
            :temperature_dependence => EffectsCoding(; base="arrhenius"),
            :thaw_depth => EffectsCoding(; base=0),
        )
    )
    display(model)
    return df_runs

end
