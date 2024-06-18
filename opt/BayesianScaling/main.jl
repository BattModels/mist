using BayesianScaling
using JSON: JSON
using DataFrames
using Turing: Turing, NUTS, MCMCThreads, sample, Chains
using MCMCChainsStorage
using HDF5: h5open
using ADTypes: AutoForwardDiff

function instantiate_model(dir)
    df = map(filter(f -> endswith(f, ".json"), readdir(dir; join=true))) do file
        JSON.parsefile(file)
    end |> DataFrame

    DataFrames.transform!(df,
        [:d_model, :n_layers] => ByRow(/) => :aspect_ratio,
        [:d_ff, :d_model] => ByRow(/) => :ff_ratio,
        # [:d_model, :n_heads] => ByRow(/) => :kv_size,
        [:num_training_steps, :eff_batch_size] => ByRow(*) => :data_size,
    )
    subset!(df,
        :min_val_loss => ByRow(x -> 1e-4 < x < 1.0),
        :steps => ByRow(x -> 2000 <= x),
        :tokenizer => ByRow(x -> x ∈ ["smirk", "ibm/MoLFormer-XL-both-10pct"]),
    )
    model = BayesianScaling.bayes_llm_scaling_model(
        df[!, :min_val_loss],
        df[!, :model_size],
        df[!, :data_size],
        df[!, :lr],
        df[!, :ff_ratio],
        df[!, :aspect_ratio]
    )
    return model, df
end

function fit_model(model)
    sampler = NUTS(1000, 0.65; init_ϵ=5.0e-3, adtype=AutoForwardDiff())
    chains = sample(model, sampler, MCMCThreads(), 100, 2; drop_warmup=true, progres=true)
    return chains
end

function main(dir)
    model, df = instantiate_model(dir)
    chains = fit_model(model)

    # Generate plots
    BayesianScaling.plot_parity(model, chains)
    BayesianScaling.plot_best_lr(chains, dfm)
    BayesianScaling.plot_scaling(chains, dfm)
    BayesianScaling.plot_scaling_parameters(chains)

    return nothing
end
