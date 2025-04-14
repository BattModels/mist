#!/usr/bin/env -S julia --color=yes --startup-file=no --project=@script --threads=auto
using DataFrames
using StatsBase
using Enzyme: Enzyme
using Dates: DateTime
using BayesianScaling: BayesianScaling, ShapedScaling, sample_chains, init_model
using Distributions: Normal, LogNormal, Exponential, truncated
using JLD2: jldopen
using Setfield: @set!

GIT_ROOT = readchomp(`git rev-parse --show-toplevel`)

include("wandb_import.jl")

function prior_d_model_lr!(priors; lr_0=1.64e-4, d_model=768, batch_size=1024)
    # LR scaling with Model size used in Attention is All You Need
    # Rescale base LR to match MoLFormer's base LR
    # Vaswani, A. et al. 2017. Attention Is All You Need. arXiv:1706.03762 [cs]. (Dec. 2017).
    @set! priors.lr.ideal.c = Normal(-0.5, 0.2)
    lr_a = lr_0 * sqrt(d_model) / sqrt(batch_size)
    @set! priors.lr.ideal.a = LogNormal(log(lr_a), 0.8)
    return priors
end

function build_models()
    formulas = Dict(
        "baseline" => ShapedScaling(),
        "additive-penalty" => ShapedScaling(geometric_penalty=false),
        "geometric-shape" => ShapedScaling(harmonic_shape_penalty=false),
        "lr-d-model" => ShapedScaling(lr_model_size=:d_model),
        "lr-d-geom-model" => ShapedScaling(lr_model_size=:d_model, harmonic_shape_penalty=false),
    )

    models = Dict()
    for eval_batch in [1, 1e3, 1e4, 1e5, 1e6]
        df = pretraining_runs(
            joinpath(GIT_ROOT, ".cache", "wandb-export");
            smoothed_eval_batch=eval_batch
        )
        subset!(df,
            :tokenizer => ByRow(==("smirk")),
            :created => ByRow(<=(DateTime(2025, 1))),
            :tags => ByRow(tags -> "dec-3-sweep" in tags),
            [:step, :max_steps] => ByRow((s, ms) -> s / ms > 0.8);
            skipmissing=true,
        )
        loss = eval_batch == 1 ? :val_loss_best : :val_loss_smooth
        df = select(df,
            :model_size,
            :d_model,
            :tokenizer,
            loss => :loss,
            [:effective_batch_size, :max_steps] => ByRow(*) => :data_size,
            :lr,
            :effective_batch_size,
            :ff_ratio, :aspect_ratio, :kv_size,
        )
        dropmissing!(df)
        subset!(df, :loss => ByRow(x -> 1e-6 < x < 1.0); skipmissing=true)

        for (name, f) in pairs(formulas)
            out = joinpath(@__DIR__, "..", "out", "dec-3-sweep-smoothed-$eval_batch--$name")
            priors = BayesianScaling.priors(f)
            if f.lr_model_size == :d_model
                priors = prior_d_model_lr!(priors)
            end
            if f.harmonic_shape_penalty
                @set! priors.ff_ratio[2] = truncated(Exponential(1e-3); upper=1e-2)
                @set! priors.kv_size[2] = truncated(Exponential(1e-3); upper=1e-2)
                @set! priors.aspect_ratio[2] = truncated(Exponential(1e-3); upper=1e-2)
            end

            model = init_model(f, df; priors)

            # Check the posterior of the prior has finite density
            θ = BayesianScaling.sample_priors(model)
            ℓ = BayesianScaling.logdensity(model, θ)
            if isinf(ℓ)
                @warn "Non-finite prior -> skipping" ℓ f
                continue
            end

            @info "Queuing Model" out model
            models[out] = deepcopy(model)
        end
    end
    return models
end

function run_models(models)
    Threads.@threads :dynamic for (out, model) in collect(pairs(models))
        @info "starting $out"
        try
            stats = @timed sample_chains(model; nchains=8, draws=1_000)
            chains = stats.value
            score = BayesianScaling.score_model(model, chains)
            perf_stats = Base.structdiff(stats, (; value=nothing, gc_stats=nothing))
            BayesianScaling.save_results(model, chains; outdir=out, score, perf_stats)
            @info "finished $out" score perf_stats
        catch e
            @error "failed $out" e
        end
    end
end

function model_summary(models)
    rows = []
    for model_dir in keys(models)
        isfile(joinpath(model_dir, "chains.jld2")) || continue
        data = jldopen(joinpath(model_dir, "chains.jld2"), "r")
        model = data["model"]
        chains = data["chains"]
        G, a, E = BayesianScaling.scaling_summary(chains)
        push!(rows, (;
            model=basename(model_dir),
            nobs=nobs(model),
            G,
            a,
            E,
            data["score"]...
        ))
    end
    df = DataFrame(rows)
    sort!(df, :waic)
    return df
end

function (@main)(args)
    models = build_models()
    run_models(models)
end
