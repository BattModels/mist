using DataFrames
using MCMCDiagnosticTools: ess_rhat
using Enzyme: Enzyme
using Dates: DateTime
using BayesianScaling: BayesianScaling, ShapedScaling, init_logdensity_model, sample_chains
using Makie: with_theme

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
        [:step, :max_steps] => ByRow((s, ms) -> s / ms > 0.8);
        skipmissing=true,
    )
    df = select(df,
        :model_size,
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

    process_model(ShapedScaling(df), df)
end
