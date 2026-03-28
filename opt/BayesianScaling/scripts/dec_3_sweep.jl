#!/usr/bin/env -S julia --color=yes --startup-file=no --project=@script --threads=auto
using DataFrames
using StatsBase
using Enzyme: Enzyme
using Dates: DateTime
using BayesianScaling: BayesianScaling, ShapedScaling, HoffmanScaling, sample_chains, init_model, pf_day
using Distributions: Normal, LogNormal, Exponential, Gamma, truncated
using JLD2: jldopen
using Setfield: @set!

GIT_ROOT = readchomp(`git rev-parse --show-toplevel`)

# include("wandb_import.jl")

function prior_d_model_lr!(priors; lr_0=1.64e-4, d_model=768, batch_size=1024)
    # LR scaling with Model size used in Attention is All You Need
    # Rescale base LR to match MoLFormer's base LR
    # Vaswani, A. et al. 2017. Attention Is All You Need. arXiv:1706.03762 [cs]. (Dec. 2017).
    @set! priors.lr.ideal.c = Normal(-0.5, 0.2)
    lr_a = lr_0 * sqrt(d_model) / sqrt(batch_size)
    @set! priors.lr.ideal.a = LogNormal(log(lr_a), 0.8)
    return priors
end

function dec_3_sweep_runs()
    df = pretraining_runs(
        joinpath(GIT_ROOT, ".cache", "wandb-export");
    )
    subset!(df,
        :tokenizer => ByRow(==("smirk")),
        :created => ByRow(<=(DateTime(2025, 1))),
        :tags => ByRow(tags -> "dec-3-sweep" in tags),
        [:step, :max_steps] => ByRow((s, ms) -> s / ms > 0.8);
        skipmissing=true,
    )
    dropmissing!(df, [:model_size, :d_model, :effective_batch_size, :max_steps, :lr, :ff_ratio, :aspect_ratio, :kv_size])
    subset!(df, :val_loss_best => ByRow(x -> 1e-6 < x < 1.0); skipmissing=true)
    return df
end

function build_models(df_sweep)
    formulas = Dict(
        "baseline" => ShapedScaling(),
        "hoffman" => HoffmanScaling(),
        "additive-penalty" => ShapedScaling(geometric_penalty=false),
        "geometric-shape" => ShapedScaling(harmonic_shape_penalty=false),
        "lr-d-model" => ShapedScaling(lr_model_size=:d_model),
        "lr-d-geom-model" => ShapedScaling(lr_model_size=:d_model, harmonic_shape_penalty=false),
    )

    models = Dict()
    for eval_batch in [1, 1e3, 1e4, 1e5, 1e6]
        df = deepcopy(df_sweep)
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
            out = joinpath(@__DIR__, "..", "out", "dec-3-sweep-smoothed-$eval_batch--$name-gamma")
            f = deepcopy(f)
            priors = BayesianScaling.priors(f)
            if f isa ShapedScaling
                if f.lr_model_size == :d_model
                    priors = prior_d_model_lr!(priors)
                end
                if f.harmonic_shape_penalty
                    @set! priors.ff_ratio[2] = truncated(Exponential(1e-3); upper=1e-2)
                    @set! priors.kv_size[2] = truncated(Exponential(1e-3); upper=1e-2)
                    @set! priors.aspect_ratio[2] = truncated(Exponential(1e-3); upper=1e-2)
                end
            end

            @set! priors.sigma = Gamma(2, 0.1)

            model = init_model(f, df; priors)

            # Check the posterior of the prior has finite density
            θ = BayesianScaling.sample_priors(model)
            ℓ = BayesianScaling.logdensity(model, θ)
            if isinf(ℓ)
                @warn "Non-finite prior -> skipping" ℓ f
                continue
            end


            @info "Queuing Model" out model
            models[out] = model
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

function fit_summary()
    rows = []
    for dir in readdir(joinpath(pkgdir(BayesianScaling), "out"); join=true)
        name = basename(dir)
        startswith(name, "dec-3-sweep") || continue
        isfile(joinpath(dir, "chains.jld2")) || continue
        data = jldopen(joinpath(dir, "chains.jld2"), "r")
        model = data["model"]
        chains = data["chains"]
        chains_scaling = haskey(chains[1, 1, :], :scaling) ? selectdim(chains, 3, :scaling) : chains
        (; G, a, E, ζ, r) = BayesianScaling.scaling_summary(chains_scaling)
        m = match(r"-([+\-e0-9\.]+)--(.*)", name)
        eff_batch_size = parse(Float64, m.captures[1])
        model_type = m.captures[2]
        eff_batch_size = eff_batch_size == 1 ? nothing : eff_batch_size

        # Track number need for full factorial
        df = DataFrame(model.observations)
        model_size = unique(df.model_size)
        data_size = unique(df.data_size)
        if "lr" in names(df)
            lr_0 = unique(@.(df.lr / sqrt(df.effective_batch_size)))
        else
            lr_0 = 1
        end
        n_fullfact = length(model_size) * length(data_size) * length(lr_0)
        fullfact_compute = sum(@.(6 * Float64(model_size) * Float64(data_size)')) * length(lr_0)
        net_compute = sum(@.(6 * Float64(df.model_size) * Float64(df.data_size))) * length(lr_0)

        push!(rows, (;
            # name,
            eff_batch_size,
            model_type,
            nobs=nobs(model),
            n_fullfact,
            fullfact_compute=fullfact_compute / pf_day,
            net_compute=net_compute / pf_day,
            n_model_size=length(model_size),
            n_data_size=length(data_size),
            lr_0=length(lr_0),
            compute_saving=net_compute / fullfact_compute,
            G,
            a,
            E,
            ζ,
            r,
            data["score"]...
        ))
    end
    df = DataFrame(rows)
    sort!(df, :waic)
    return df
end

function export_runs()
    df_sweep = dec_3_sweep_runs()
    outdir = joinpath(pkgdir(BayesianScaling), "out")
    mkpath(outdir)
    open(joinpath(outdir, "dec_3_sweep_runs.jsonl"), "w") do fid
        for run in eachrow(df_sweep)
            println(fid, JSON.json(Dict(pairs(run))))
        end
    end
    return df_sweep
end

function (@main)(args)
    # Record runs
    df_sweep = export_runs()

    # Build models
    models = build_models(df_sweep)
    run_models(models)
end
