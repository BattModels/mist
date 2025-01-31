using BayesianScaling
using BayesianScaling: find, pretraining_runs
using JSON: JSON
using DataFrames
using DynamicHMC: mcmc_with_warmup, ProgressMeterReport, NoProgressReport
using HDF5: h5open
using StatsBase: mean_and_std, mean
using UUIDs: uuid4
using Random
using Enzyme
using JLD2: jldsave
using ADTypes: AutoForwardDiff
using CairoMakie

RUNS_DIR = abspath(joinpath(pathof(BayesianScaling), "..", "..", "..", "..", ".cache", "wandb-export"))

function figure_training_campaign(dir=RUNS_DIR)
    f = Figure()
    ax = Axis(f[1, 1];
        xscale=log10,
        yscale=log10,
        xlabel="Compute [FLOP]",
        ylabel="Validation Loss",
    )
    df = pretraining_runs(dir)
    df = sort!(df, :created)
    C = @. 6 * df.model_size * df.step * df.effective_batch_size
    scatter!(ax, C, df.val_loss_best)
    @info "Cummulative Compute Budget" sum(C; init=0.0)
    ax_cum = Axis(f[1, 2];
        yscale=log10,
        ylabel="Cummulative Compute Spend [FLOP]"
    )
    lines!(ax_cum, df.created, accumulate(+, C; init=0.0))
    f
end

function init_hoffman(df=pretraining_runs(), tokenizer="smirk")
    df = select(df,
        :model_size,
        :tokenizer,
        :val_loss_best => :loss,
        [:effective_batch_size, :step] => ByRow(*) => :data_size,
    )
    dropmissing!(df)
    df = subset(df,
        :loss => ByRow(x -> 1e-4 < x < 1.0),
        :tokenizer => ByRow(==(tokenizer)),
    )
    m = BayesianScaling.HoffmanScaling(df)
    m = BayesianScaling.init_logdensity_model(m, :ForwardDiff)
    return m, df
end

function init_shaped(df=pretraining_runs(), tokenizer="smirk")
    df = select(df,
        :model_size,
        :tokenizer,
        :val_loss_best => :loss,
        [:effective_batch_size, :step] => ByRow(*) => :data_size,
        :lr,
        :effective_batch_size,
        :ff_ratio, :aspect_ratio, :kv_size,
        :optimizer,
        :beta1, :beta2,
    )
    dropmissing!(df)
    df = subset(df,
        :loss => ByRow(x -> 1e-4 < x < 1.0),
        :tokenizer => ByRow(==(tokenizer)),
        :optimizer => ByRow(==("deepspeed.ops.lamb.FusedLamb")),
    )
    df.lr_base = @. df.lr / sqrt(df.effective_batch_size / 1024)
    m = BayesianScaling.ShapedScaling(df)
    m = BayesianScaling.init_logdensity_model(m, :Enzyme)
    return m, df
end


function init_training_progress(df::DataFrame)
    df = clean_dataset(df)
    DataFrames.transform!(df,
        [Symbol("val/loss_epoch"), :num_training_steps] => ByRow(rel_loss_trace) => [:rel_step_trace, :val_loss_trace],
    )
    dropmissing!(df)
    subset!(df,
        :min_val_loss => ByRow(x -> 1e-4 < x < 1.0),
        :tokenizer => ByRow(x -> x ∈ ["smirk",]);
    )

    model = BayesianScaling.TrainingProgress(df)
    model = BayesianScaling.init_logdensity_model(model, :Enzyme)
    return model, df
end


function rel_loss_trace(x::Dict, num_training_steps, warmup_steps=2000)
    rel_step = x["step"] ./ num_training_steps
    n_warm = findfirst(>(warmup_steps), x["step"])
    isnothing(n_warm) && return missing, missing
    loss = float.(x["loss"])
    return rel_step[n_warm:end], loss[n_warm:end]
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

function process_model(model, df)
    outdir = mkpath(joinpath(@__DIR__, "out", string(uuid4())) * "/")
    @info "Will save output to $outdir"

    # Sample chaings
    y, yr = BayesianScaling.sample_chains(model)
    @info "Finished sampling"
    jldsave(joinpath(outdir, "chains.jld2"); model, df, chains=y, chains_raw=yr)

    # Check convergence
    @info "generating figures"
    save_figures(outdir, model, df, y, yr)

end

function save_figures(outdir, model, df, chains, raw_chains)
    # Subsample for faster plotting
    mkpath(outdir)
    chains = BayesianScaling.subsample(chains, 100)

    # Generate Plots
    θ_scaling = selectdim(chains, 3, :scaling)
    @sync begin
        Threads.@spawn savefig(outdir, "scaling.pdf", BayesianScaling.plot_scaling(chains, df))
        Threads.@spawn savefig(outdir, "lr_map.pdf", BayesianScaling.plot_lr_map(model.ℓ.log_density_function, chains, df))
        Threads.@spawn savefig(outdir, "compute_optimal.pdf", BayesianScaling.plot_compute_optimal(θ_scaling, df))
        Threads.@spawn savefig(outdir, "penalties.pdf", BayesianScaling.plot_penalty(model.ℓ.log_density_function, chains))
        Threads.@spawn savefig(outdir, "summary.pdf", BayesianScaling.figure_ai4x(model.ℓ.log_density_function, chains))

        # Bayesian Plots
        for sym in [:scaling, :lr]
            θ = selectdim(raw_chains, 3, sym)
            Threads.@spawn savefig(outdir, "raw_chains_$sym.pdf", BayesianScaling.plot_chains(θ))
            Threads.@spawn savefig(outdir, "raw_chains_covar_$sym.pdf", BayesianScaling.plot_chain_covariance(θ))
        end
    end

    return outdir
end

