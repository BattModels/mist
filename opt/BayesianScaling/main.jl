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
    chains = sample(model, sampler, MCMCThreads(), 10_000, 128; drop_warmup=true, progres=true)
    return chains
end

savefig(name::String, f) = save(joinpath(@__DIR__, "figs", name), f)
savefig(dir::String, name::String, f) = save(joinpath(dir, name), f)

function main(args)
    dir = args[1]
    outdir = mkpath(joinpath(@__DIR__, "out", string(uuid4())) * "/")

    @info "Instantiating model"
    model, df = instantiate_model(dir)
    open(joinpath(outdir, "dataset.json"), "w") do fid
        JSON.print(fid, eachrow(df))
    end

    @info "Fitting model"
    chains = fit_model(model)
    h5open(joinpath(outdir, "chains.h5"), "w") do h5
        write(h5, chains)
    end

    # Generate plots
    @info "Generating plots"
    savefig(outdir, "parity.pdf", BayesianScaling.plot_parity(model, chains))
    savefig(outdir, "scaling.pdf", BayesianScaling.plot_scaling(chains, df))
    savefig(outdir, "lr_scaling.pdf", BayesianScaling.plot_best_lr(chains, df))
    savefig(outdir, "chains.pdf", BayesianScaling.plot_chains(model, chains))
    savefig(outdir, "chain_cov.pdf", BayesianScaling.plot_chain_covariance(chains, model))

    return 0
end

!isinteractive() && exit(main(ARGS))
