using BayesianScaling
using JSON: JSON
using DataFrames
using Turing: Turing, NUTS, MCMCThreads, sample, Chains
using MCMCChainsStorage
using HDF5: h5open
using ADTypes: AutoForwardDiff
using StatsBase: mean_and_std, mean

@static isinteractive() ? using GLMakie : using CairoMakie

function load_dataset(dir)
    row = map(filter(f -> endswith(f, ".json"), readdir(dir; join=true))) do file
        JSON.parsefile(file; null=missing)
    end
    df_cols = unique(Iterators.flatten(keys.(row)))
    filter!(r -> keys(r) == Set(df_cols), row)
    return DataFrame(row)
end

instantiate_model(dir::String) = instantiate_model(load_dataset(dir))
function instantiate_model(df::DataFrame)
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
    subset!(df, "val/loss_epoch" => ByRow(x -> detect_spike(x) < 1e-4); skipmissing=true)
    subset!(df, "val/loss_epoch" => ByRow(x -> max_noise_level(x) <= 10); skipmissing=true)
    dropmissing!(df)

    DataFrames.transform!(df,
        [:d_model, :d_ff, :n_layers] => ByRow(BayesianScaling.non_embedding_size) => :model_size,
        [:d_model, :n_layers] => ByRow(/) => :aspect_ratio,
        [:d_ff, :d_model] => ByRow(/) => :ff_ratio,
        [:d_model, :n_heads] => ByRow(/) => :kv_size,
        [:num_training_steps, :eff_batch_size] => ByRow(*) => :data_size,
        [:steps, :num_training_steps] => ByRow(relative_steps) => :relative_step,
        ["val/loss_epoch", :num_training_steps] => ByRow(rel_loss_trace) => :val_loss_trace,
    )
    subset!(df,
        :min_val_loss => ByRow(x -> 1e-4 < x < 1.0),
        :steps => ByRow(x -> 2000 <= x),
        # [:steps, :num_training_steps] => ByRow((x, y) -> x == y -1),
        :tokenizer => ByRow(x -> x ∈ ["smirk",]);
    )
    # df = Iterators.flatmap(eachrow(df)) do row
    #     base_row = Dict(pairs(row))
    #     trace = pop!(base_row, Symbol("val/loss_epoch"))
    #     map(zip(trace["step"], trace["loss"])) do (step, loss)
    #         row = deepcopy(base_row)
    #         row[:steps] = step
    #         row[:min_val_loss] = loss
    #         row
    #     end
    # end |> DataFrame
    model = BayesianScaling.bayes_llm_scaling_model(
        df[!, :min_val_loss],
        df[!, :model_size],
        df[!, :data_size],
        df[!, :lr],
        df[!, :ff_ratio],
        df[!, :aspect_ratio],
    )
    return model, df
end

function init_training_progress(df::DataFrame)
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
    subset!(df, "val/loss_epoch" => ByRow(x -> detect_spike(x) < 1e-4); skipmissing=true)
    subset!(df, "val/loss_epoch" => ByRow(x -> max_noise_level(x) <= 10); skipmissing=true)
    dropmissing!(df)

    DataFrames.transform!(df,
        [:d_model, :d_ff, :n_layers] => ByRow(BayesianScaling.non_embedding_size) => :model_size,
        [:d_model, :n_layers] => ByRow(/) => :aspect_ratio,
        [:d_ff, :d_model] => ByRow(/) => :ff_ratio,
        [:d_model, :n_heads] => ByRow(/) => :kv_size,
        [:num_training_steps, :eff_batch_size] => ByRow(*) => :data_size,
        [:steps, :num_training_steps] => ByRow(relative_steps) => :relative_step,
        [Symbol("val/loss_epoch"), :num_training_steps] => ByRow(rel_loss_trace) => [:rel_step_trace, :val_loss_trace],
    )
    dropmissing!(df)
    subset!(df,
        :min_val_loss => ByRow(x -> 1e-4 < x < 1.0),
        :steps => ByRow(x -> 2000 <= x),
        # [:steps, :num_training_steps] => ByRow((x, y) -> x == y -1),
        :tokenizer => ByRow(x -> x ∈ ["smirk",]);
    )

    model = BayesianScaling.training_progress(
        df[!, :val_loss_trace],
        df[!, :rel_step_trace],
        df[!, :model_size],
        df[!, :data_size],
        df[!, :lr],
        df[!, :ff_ratio],
        df[!, :aspect_ratio],
        df[!, :eff_batch_size],
    )
    return model, df
end


function rel_loss_trace(x::Dict, num_training_steps, warmup_steps=2000)
    rel_step = relative_steps.(x["step"], num_training_steps, warmup_steps)
    n_warm = findfirst(>(0), rel_step)
    isnothing(n_warm) && return missing, missing
    return rel_step[n_warm:end], float.(x["loss"][n_warm:end])
    # return Int.(x["step"]), float.(x["loss"])
end

function relative_steps(step, num_training_steps, warmup_steps=2000)
    (step - warmup_steps) / (num_training_steps - warmup_steps)
end

detect_spike(x::Dict) = detect_spike(x["step"], x["loss"])
function detect_spike(step, loss)
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

max_noise_level(x::Dict; kwargs...) = max_noise_level(x["step"], x["loss"]; kwargs...)
function max_noise_level(steps, loss; n=20)
    noise = 0.0
    for i in 1:(length(loss)-n)
        mu, sigma = mean_and_std(log10.(loss[i:i+n]))
        noise = max(noise, sigma / mu)
    end
    return noise
end

function fit_model(model)
    sampler = NUTS(1000, 0.65; init_ϵ=5.0e-3, adtype=AutoForwardDiff())
    chains = sample(model, sampler, MCMCThreads(), 10, 4; drop_warmup=true, progres=true)
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

function plot_traces(df)
    f = Figure()
    ax = Axis(f[1, 1];
        xscale=log10, yscale=log10,
        limits=((1, nothing), (1e-4, 4.0)),
    )
    idx = Observable(1)
    sl = Slider(f[2, 1], range = 1:nrow(df), startvalue = 1)
    trace = lift(sl.value) do idx
        trace = df[idx, "val/loss_epoch"]
        Point2f.(trace["step"], trace["loss"])
    end
    lines!(ax, trace)
    train_trace = lift(sl.value) do idx
        trace = df[idx, "train/loss_step"]
        Point2f.(trace["step"], trace["loss"])
    end
    lines!(ax, train_trace)


    vel = lift(sl.value) do idx
        vel_trace = df[idx, "val/loss_epoch"]
        step = vel_trace["step"]
        loss = vel_trace["loss"]
        vel = diff(loss) ./ diff(step)
        accel = diff(vel) ./ diff(step)[1:end-1]
        x = step[1:end-2]
        Point2f.(x, accel)
    end

    ax2 = Axis(f[1, 2]; xscale=log10)
    lines!(ax2, vel)
    linkxaxes!(ax, ax2)
    return f
end

function plot_loss_curve(df, chains)
    f = Figure()
    ax = Axis(f[1, 1]; yscale=log10, xscale=log10)
    for row in eachrow(df)
        step = row["val/loss_epoch"]["step"]
        n_warm = findfirst(>=(2000), step)
        isnothing(n_warm) && continue
        ns = row.num_training_steps
        rel_step = @. (step - n_warm) / (ns - n_warm)
        expected_loss = mean(BayesianScaling.hoffman_scaling(row.model_size, row.data_size, chains))
        loss = row["val/loss_epoch"]["loss"] ./ expected_loss
        scatter!(ax, rel_step[n_warm:end], loss[n_warm:end]; marker=:+, markersize=3)
    end
    return f
end

!isinteractive() && exit(main(ARGS))
