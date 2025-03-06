using DataFrames
using Enzyme: Enzyme
using Dates: DateTime
using BayesianScaling: BayesianScaling, ShapedScaling, sample_chains, init_model
using Distributions: Normal, LogNormal
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

formulas = Dict(
    "baseline" => ShapedScaling(),
    "additive-penalty" => ShapedScaling(geometric_penalty=false),
    "geometric-shape" => ShapedScaling(harmonic_shape_penalty=false),
    "lr-d-model" => ShapedScaling(lr_model_size=:d_model),
)

models = Dict()
for eval_batch in [1e3, 1e4, 1e5, 1e6]
    df = pretraining_runs(
        joinpath(GIT_ROOT, ".cache", "wandb-export");
        smoothed_eval_batch=eval_batch
    )
    subset!(df,
        :tokenizer => ByRow(==("smirk")),
        :created => ByRow(<=(DateTime(2025, 1))),
        :tags => ByRow(tags -> "dec-3-sweep" in tags),
        [:step, :max_steps] => ByRow((s, ms) -> s / ms > 0.8);
        :loss => ByRow(x -> 1e-6 < x < 1.0),
        skipmissing=true,
    )
    df = select(df,
        :model_size,
        :d_model,
        :tokenizer,
        :val_loss_smooth => :loss,
        [:effective_batch_size, :max_steps] => ByRow(*) => :data_size,
        :lr,
        :effective_batch_size,
        :ff_ratio, :aspect_ratio, :kv_size,
    )
    dropmissing!(df)

    for (name, f) in pairs(formulas)
        out = joinpath(@__DIR__, "..", "out", "dec-3-sweep-smoothed-$eval_batch--$name")
        priors = BayesianScaling.priors(f)
        if f.lr_model_size == :d_model
            priors = prior_d_model_lr!(priors)
        end

        model = init_model(f, df; priors)
        @info "Queing Model" out model
        models[out] = model
    end
end

Threads.@threads :dynamic for (out, model) in pairs(models)
    stats = @timed sample_chains(model; nchains=8, draws=10_000)
    chains = stats.value
    score = BayesianScaling.score_model(model, chains)
    perf_stats = Base.structdiff(stats, (; value=nothing, gc_stats=nothing))
    BayesianScaling.save_results(model, chains; outdir=out, score, perf_stats)
    @info "finished $out" score perf_stats
end
