using BayesianScaling
using JSON: JSON
using DataFrames
using DynamicHMC: mcmc_with_warmup, ProgressMeterReport, NoProgressReport
using HDF5: h5open
using StatsBase: mean_and_std, mean
using UUIDs: uuid4
using Interpolations: linear_interpolation
using Random
using Enzyme
using JLD2

#@static isinteractive() ? using GLMakie : using CairoMakie

function load_dataset(dir)
    row = map(filter(f -> endswith(f, ".json"), readdir(dir; join=true))) do file
        JSON.parsefile(file; null=missing)
    end
    df_cols = unique(Iterators.flatten(keys.(row)))
    filter!(r -> keys(r) == Set(df_cols), row)
    return DataFrame(row)
end

instantiate_model(dir::String) = instantiate_model(load_dataset(dir))

function clean_dataset(df::DataFrame)
    df = DataFrames.select(df,
        :id,
        :lr,
        :min_val_loss, :min_train_loss,
        :steps,
        :gas, :eff_batch_size, :macro_batch_size,
        :tokenizer,
        :d_model, :d_ff, :n_layers, :n_heads,
        :num_training_steps,
        "val/loss_epoch",
    )
    subset!(df, "val/loss_epoch" => ByRow(x -> gradient(x["step"], x["loss"]) < 1e-4); skipmissing=true)
    subset!(df, "val/loss_epoch" => ByRow(x -> max_noise_level(x["loss"]) <= 10); skipmissing=true)
    dropmissing!(df)

    DataFrames.transform!(df,
        [:d_model, :d_ff, :n_layers] => ByRow(BayesianScaling.non_embedding_size) => :model_size,
        [:d_model, :n_layers] => ByRow(/) => :aspect_ratio,
        [:d_ff, :d_model] => ByRow(/) => :ff_ratio,
        [:d_model, :n_heads] => ByRow(/) => :kv_size,
        [:num_training_steps, :eff_batch_size] => ByRow(*) => :data_size,
        Symbol("val/loss_epoch") => ByRow(x -> minimum(x["loss"])) => :loss
    )
    subset!(df,
        :min_val_loss => ByRow(x -> 1e-4 < x < 1.0),
        :steps => ByRow(x -> 2000 <= x),
    )
end

function init_hoffman(df::DataFrame)
    df = clean_dataset(df)
    df = DataFrames.subset!(df,
        [:steps, :num_training_steps] => ByRow((x, y) -> x == y - 1),
        :tokenizer => ByRow(x -> x ∈ ["smirk",]);
    )
    dropmissing!(df)
    model = BayesianScaling.HoffmanScaling(df)
    model = BayesianScaling.init_logdensity_model(model, :ForwardDiff)
    return model, df
end

function init_training_progress(df::DataFrame)
    df = clean_dataset(df)
    DataFrames.transform!(df,
        [Symbol("val/loss_epoch"), :num_training_steps] => ByRow(rel_loss_trace) => [:rel_step_trace, :val_loss_trace],
    )
    dropmissing!(df)
    subset!(df,
        :min_val_loss => ByRow(x -> 1e-4 < x < 1.0),
        :steps => ByRow(x -> 2000 <= x),
        :tokenizer => ByRow(x -> x ∈ ["smirk",]);
    )

    model = BayesianScaling.TrainingProgress(df)
    model = BayesianScaling.init_logdensity_model(model, :Enzyme)
    return model, df
end


function rel_loss_trace(x::Dict, num_training_steps, warmup_steps=2000)
    rel_step = relative_steps.(x["step"], num_training_steps, warmup_steps)
    n_warm = findfirst(>(0), rel_step)
    isnothing(n_warm) && return missing, missing
    loss = float.(x["loss"])
    itp = linear_interpolation(rel_step[n_warm:end], loss[n_warm:end])
    step = range(rel_step[n_warm], rel_step[end], length=100)
    return step, itp.(step)
end

function relative_steps(step, num_training_steps, warmup_steps=2000)
    (step - warmup_steps) / (num_training_steps - warmup_steps)
end

function gradient(step, loss)
    length(step) < 3 && return missing
    f_b = loss[1:end-2]
    f_c = loss[2:end-1]
    f_f = loss[3:end]
    h = diff(step)
    h_b = h[1:end-1]
    h_f = h[2:end]
    accel = @. (f_f - 2f_c + f_b) / (h_b * h_f)
    return maximum(abs.(accel))
end

"""
    max_noise_level(x; n=20)

Compute a moving window of the noise level (σ/μ) over `x`
"""
function max_noise_level(x; n=20)
    noise = 0.0
    for i in 1:(length(x)-n)
        mu, sigma = mean_and_std(log10.(x[i:i+n]))
        noise = max(noise, sigma / mu)
    end
    return noise
end

savefig(dir::String, name::String, f) = save(joinpath(dir, name), f)

function main(args)
    dir = args[1]
    outdir = mkpath(joinpath(@__DIR__, "out", string(uuid4())) * "/")
    @info "Saving to $outdir"

    @info "Instantiating model"
    df = load_dataset(dir)
    model, df = init_training_progress(df)
    open(joinpath(outdir, "dataset.json"), "w") do fid
        JSON.print(fid, eachrow(df))
    end

    @info "Fitting model with $(Threads.nthreads()) chains"
    Threads.@threads for i in 1:Threads.nthreads()
        local_model = deepcopy(model)
        results = mcmc_with_warmup(Random.default_rng(), local_model, 10_000; reporter=NoProgressReport())
        posterior = BayesianScaling.transform_samples(local_model, results.posterior_matrix)
        jldsave(joinpath(outdir, "chain-$i.jld2"); model=local_model, posterior, results...)
        @info "finished chain $i"
    end
    return 0

    # Generate plots
    @info "Generating plots"
    savefig(outdir, "parity.pdf", BayesianScaling.plot_parity(model, chains))
    savefig(outdir, "scaling.pdf", BayesianScaling.plot_scaling(chains, df))
    savefig(outdir, "lr_scaling.pdf", BayesianScaling.plot_best_lr(chains, df))
    savefig(outdir, "chains.pdf", BayesianScaling.plot_chains(model, chains))
    savefig(outdir, "chain_cov.pdf", BayesianScaling.plot_chain_covariance(model, chains))

    return 0
end



!isinteractive() && exit(main(ARGS))
