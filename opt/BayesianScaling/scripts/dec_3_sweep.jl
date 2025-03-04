using DataFrames
using MCMCDiagnosticTools: ess_rhat
using Enzyme: Enzyme
using Dates: DateTime
using BayesianScaling: BayesianScaling, ShapedScaling, init_logdensity_model, sample_chains
using Distributions: Normal, LogNormal
using Makie: with_theme
using Setfield: @set!

GIT_ROOT = readchomp(`git rev-parse --show-toplevel`)

include("wandb_import.jl")
include("utils.jl")

Threads.@threads :dynamic for eval_batch in [1e3, 1e4, 1e5, 1e6]
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
    df = select(df,
        :model_size,
        :d_model,
        :tokenizer,
        :val_loss_smooth => :loss,
        [:effective_batch_size, :max_steps] => ByRow(*) => :data_size,
        :lr,
        :effective_batch_size,
        :ff_ratio, :aspect_ratio, :kv_size,
        :optimizer,
        :beta1, :beta2,
    )
    dropmissing!(df)
    df = subset(df,
        :loss => ByRow(x -> 1e-6 < x < 1.0),
        :tokenizer => ByRow(==("smirk")),
    )
    m = ShapedScaling(df)

    # LR scaling with Model size used in Attention is All You Need
    # Rescale base LR to match MoLFormer's base LR
    # Vaswani, A. et al. 2017. Attention Is All You Need. arXiv:1706.03762 [cs]. (Dec. 2017).
    df.model_size_lr = df.d_model
    @set! m.priors.lr.ideal.c = Normal(-0.5, 0.2)
    lr_a = 1.64e-4 * sqrt(768) / sqrt(1024)
    @set! m.priors.lr.ideal.a = LogNormal(log(lr_a), 0.8)

    outdir = process_model(m, df)
    @info "saved" eval_batch outdir
end
