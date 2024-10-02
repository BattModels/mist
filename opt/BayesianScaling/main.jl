using BayesianScaling
using JSON: JSON
using DataFrames
using DynamicHMC: mcmc_with_warmup, ProgressMeterReport, NoProgressReport
using Dates: Dates, DateTime
using HDF5: h5open
using StatsBase: mean_and_std, mean
using UUIDs: uuid4
using Random
using Enzyme
using JLD2
using ADTypes: AutoForwardDiff

@static isinteractive() ? using GLMakie : using CairoMakie

RUNS_DIR=abspath(joinpath(pathof(BayesianScaling), "..", "..", "..", "..", ".cache", "wandb-export"))

function find(dir, pattern)
    found = String[]
    for (root, dirs, files) in walkdir(dir)
        for file in files
            path = joinpath(root, file)
            if match(pattern, path) !== nothing
                push!(found, path)
            end
        end
    end
    return found
end

function pretraining_runs(dir=RUNS_DIR)
    row = []
    for file in find(joinpath(dir, "pretraining"), r".*\.json")
        run = JSON.parsefile(file; null=missing)
        d_model=run["model"]["d_model"]
        ff_ratio=run["model"]["d_ff"] / d_model
        kv_size= Int(run["model"]["d_model"] // run["model"]["n_heads"])
        aspect_ratio = d_model / run["model"]["n_layers"]
        beta = run["optimizer"]["betas"]
        beta1 = !ismissing(beta) ? beta[1] : missing
        beta2 = !ismissing(beta) ? beta[2] : missing
        push!(row, (;
            id=run["id"],
            state=run["state"],
            user=run["user"],
            cluster=run["cluster"],
            commit=run["commit"],
            created=DateTime(run["created"][1:23], Dates.ISODateTimeFormat),
            model_size=run["model"]["model_size"],
            d_model,
            ff_ratio,
            kv_size,
            aspect_ratio,
            optimizer=run["optimizer"]["class_path"],
            tokenizer=run["data"]["tokenizer"],
            max_steps=run["trainer"]["num_training_steps"],
            step=run["trainer"]["step"],
            tokens=run["trainer"]["tokens"],
            masked_tokens=run["trainer"]["masked_tokens"],
            effective_batch_size=run["trainer"]["effective_batch_size"],
            lr=run["optimizer"]["lr"],
            beta1,
            beta2,
            val_loss_best=run["metrics"]["val_loss_best"],
            val_loss_last=run["metrics"]["val_loss_last"],
        ))
    end
    return DataFrame(row)
end

function figure_training_campaign(dir=RUNS_DIR)
    f = Figure()
    ax = Axis(f[1,1];
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
    ax_cum = Axis(f[1,2];
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
        [:effective_batch_size, :step] => ByRow(*) => :data_size
    )
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
    )
    dropmissing!(df)
    df = subset(df,
        :loss => ByRow(x -> 1e-4 < x < 1.0),
        :tokenizer => ByRow(==(tokenizer)),
        :optimizer => ByRow(==("deepspeed.ops.lamb.FusedLamb")),
    )
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

function main(args)
    dir = args[1]
    outdir = mkpath(joinpath(@__DIR__, "out", string(uuid4())) * "/")
    @info "Will save output to $outdir"

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

    @info "Done"
    return 0
end



!isinteractive() && exit(main(ARGS))
