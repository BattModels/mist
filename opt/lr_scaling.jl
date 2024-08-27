using Turing: sample, @model, MLE, NUTS
using Distributions: fit, Normal, Uniform, LogNormal, MvLogNormal, Exponential
using MLUtils: splitobs
using DataFrames
using JSON: JSON
using LinearAlgebra: I
using Optim: optimize
using Statistics: mean, std, median
using StatsBase: quantile, sample
using GLMakie
using Random: shuffle!

function logrange(lb, ub; base=10, kwargs...)
    iter = range(log(base, lb), log(base, ub); kwargs...)
    if base == 10
        f = exp10
    elseif base == 2
        f = exp2
    else
        f = x -> base^x
    end
    return Iterators.map(f, iter)
end


@model function llm_loss(loss, x)
    # Base LLM Scaling model from: Hoffmann, J. et al. 2022.
    # Training Compute-Optimal Large Language Models. arXiv.
    A ~ LogNormal(log(500), 2)
    B ~ LogNormal(log(500), 2)
    α ~ Uniform(0, 2)
    β ~ Uniform(0, 2)
    E ~ LogNormal(-2, 3)
    σ² ~ Exponential(1)

    # Unpack parameters
    model_size = x[:, 1]
    data_size = x[:, 2]
    lr = x[:, 3]
    ff_ratio = x[:, 4]
    aspect_ratio = x[:, 5]

    # Estimate Loss
    mu = llm_scaling(model_size, data_size; A, B, α, β, E)

    # LR Scaling Model
    lr_0 ~ LogNormal(log(5e-5), 2)  # Optimal LR at N = 1
    lr_n ~ LogNormal(log(1e-2), 4)  # Scaling of lr with log N
    lr_p ~ LogNormal(0, log(5))     # Penalty term for deviation from lr_eff
    log_lr_eff = log_lr(model_size; lr_0, lr_n)
    @. mu += lr_p * (log(lr) - log_lr_eff)^2

    # Shape Factors
    ff_ratio_0 ~ LogNormal(log(1), 2)   # Ideal ff_ratio
    ff_ratio_p ~ LogNormal(0, 2)        # Penalty term for deviation
    aspect_ratio_0 ~ LogNormal(log(64), 2)
    aspect_ratio_p ~ LogNormal(0, 2)
    @. mu += ff_ratio_p * logsqdev(ff_ratio, ff_ratio_0)
    @. mu += aspect_ratio_p * logsqdev(aspect_ratio, aspect_ratio_0)

    return loss ~ MvLogNormal(log.(mu), σ² * I)
end

# Estimate the loss of a llm
function llm_scaling(model_size, data_size; A, B, α, β, E, kwargs...)
    return @. (A / (model_size^α)) + (B / (data_size^β)) + E
end

logsqdev(x, y) = (log(x) - log(y))^2

function shape_loss(ff_ratio, aspect_ratio; ff_ratio_0, ff_ratio_p, aspect_ratio_0, aspect_ratio_p, kwargs...)
    return @. ff_ratio_p * logsqdev(ff_ratio, ff_ratio_0) + aspect_ratio_p * logsqdev(aspect_ratio, aspect_ratio_0)
end

function compute_optimal_model(flops; A, α, B, β, kwargs...)
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    return @. G * (flops / 6)^a
end

function optimal_dataset_for_model(N; A, α, B, β, kwargs...)
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    @. (N^(1 / a - 1)) * G^(-1 / a)
end

# Estimate the optimal log(lr) for a given model size
log_lr(N; lr_0, lr_n, kwargs...) = @. lr_0 - (1 - lr_n) * log(N)

function sample_posterior(f, chains, args...)
    x = get(chains, chains.name_map.parameters)
    return f(args...; x...) |> Array
end

function load_data(dir)
    return map(filter(f -> endswith(f, ".json"), readdir(dir; join=true))) do file
        JSON.parsefile(file)
    end |> DataFrame
end

function fit_models(df)
    df = transform(df,
        [:d_model, :n_layers] => ByRow(/) => :aspect_ratio,
        [:d_ff, :d_model] => ByRow(/) => :ff_ratio,
        # [:d_model, :n_heads] => ByRow(/) => :kv_size,
        [:steps, :macro_batch_size] => ByRow(*) => :data_size,
    )
    df = subset(df,
        :min_val_loss => ByRow(x -> 1e-3 < x),
        :tokenizer => ByRow(x -> x ∈ ["smirk", "ibm/MoLFormer-XL-both-10pct"]),
    )
    y = df.min_val_loss
    x = Matrix(select(df, :model_size, :data_size, :lr, :ff_ratio, :aspect_ratio))
    model = llm_loss(y, x)
    mle_fit = optimize(model, MLE())
    chains = sample(model, NUTS(1000, 0.65; init_ϵ=0.001), 1_000;
        initial_params=Vector(mle_fit.values),
        drop_warmup=true,
        progres=true
    )
    return chains, df
end

@recipe(PredictionBand, x, center, lower, upper) do scene
    Theme(
        color=:black,
        band_color=(:blue, 0.4),
    )
end

function Makie.plot!(plt::PredictionBand)
    band!(plt, plt.x, plt.lower, plt.upper; color=plt.band_color)
    lines!(plt, plt.x, plt.center; color=plt.color)
    return plt
end

function plot_best_lr(post_model, df; N=range(1e6, 1e9; length=100))
    f = Figure()


    # Prediction Plot
    gl = GridLayout(f[1:2, 1])
    ax = Axis(gl[1, 1];
        yscale=log10,
        xscale=log10,
        limits=(extrema(N), (5e-6, 1e-3)),
        ylabel="Learning Rate, η",
        xlabel="Model Size (Non-Embedding)",
    )
    s = exp.(sample_posterior(log_lr, post_model, N'))
    q = map(c -> quantile(c, (0.5, 0.025, 0.975)), eachcol(s))
    predictionband!(ax, N, [l[1] for l in q], [l[2] for l in q], [l[3] for l in q];
        color=:black,
        band_color=(:slategray, 0.2),
        label="95% Prediction Interval for η"
    )

    # Add measured points
    h = scatter!(ax, df.model_size, df.lr; color=df.min_val_loss, colorscale=log10)
    Colorbar(gl[1, 2], h; label="Validation Loss")
    scatter!(ax, 29491200, 1.6e-4; marker=:star5, color=:red, markersize=15, label="MolFormer")
    axislegend(ax; position=:rb)

    return f
end

function plot_scaling(post_model, df; N=logrange(1e4, 1e10; length=50), C=logrange(1e12, 1e24; length=50))
    f = Figure()
    N = collect(N)
    C = collect(C)
    ax = Axis(f[1, 1];
        yscale=log10,
        xscale=log10,
        limits=(extrema(C), extrema(N)),
        xlabel="Compute Budget (FLOPs)",
        ylabel="Model size",
    )

    loss = Matrix{Float32}(undef, length(C), length(N))
    for (idx, n) in enumerate(N)
        d = @. C / (6n)
        loss[:, idx] = median(sample_posterior(llm_scaling, post_model, n, d'); dims=1)
    end
    loss_low = min(minimum(df.min_val_loss), minimum(loss))
    loss_high = maximum(df.min_val_loss)
    h = contourf!(ax, C, N, loss;
        levels=collect(logrange(loss_low, loss_high; length=20)),
        colorscale=log10,
    )
    # Colorbar(f[1, 2], h; label="Validation Loss")


    # Add Empirical Loss
    df.flops = @. 6 * df.model_size * df.data_size
    h = scatter!(ax, df.flops, df.model_size;
        color=df.min_val_loss,
        colormap=h.colormap,
        colorrange=@lift(extrema($(h.levels))),
        colorscale=h.colorscale,
        strokewidth=0.5,
        strokecolor=:black,
        label="Emperical Data"
    )
    Colorbar(f[1, 2], h; label="Validation Loss")

    # Add Compute Optimal Frontier
    s = sample_posterior(compute_optimal_model, post_model, C')
    q = map(c -> quantile(c, (0.5, 0.025, 0.975)), eachcol(s))
    predictionband!(ax, C, [x[1] for x in q], [x[2] for x in q], [x[3] for x in q];
        color=:black,
        band_color=(:slategray, 0.4),
        label="Compute Optimal Frontier ±25%",
    )
    hlines!(ax, 29491200; label="MolFormer", color=:red)
    vlines!(ax, 60 * 60 * 44e15; label="Polaris-hour", color=:purple)
    vlines!(ax, 60 * 60 * 2.5e12; label="Artemis-hour", color=:blue)

    f[2, 1] = Legend(f, ax; tellheight=true, tellwidth=false, orientation=:horizontal, nbanks=2)


    return f
end

function plan_lr_sweep(post_model, N, p::AbstractRange=range(0.025, 0.975; length=5))
    s = exp.(sample_posterior(log_lr, post_model, N))
    d = fit(LogNormal, s)
    eta = quantile(d, p)
    eta = round.(eta; sigdigits=2)
    return unique(eta)
end

function plan_data_sweep(post_model, N; n=5, l=0.025)
    D_optimal = fit(LogNormal, sample_posterior(optimal_dataset_for_model, post_model, N))
    p = range(l, 1 - l; length=n)
    D_proposed = quantile(D_optimal, p)
end

function plan_config(d_model, post_model; kv_size=64)
    @assert d_model % kv_size == 0
    @show d_ff = d_model .* [1, 2, 4]
    @show n_heads = d_ff ./ kv_size .|> Int
    @show n_layers = unique(fld.(d_model, 50:5:130)) .|> Int
    df = allcombinations(DataFrame; d_model, d_ff, n_heads, n_layers)
    transform!(df,
        [:d_model, :d_ff, :n_layers] => ByRow(model_size) => :N,
        [:d_ff, :d_model] => ByRow(/) => :ff_ratio,
        [:d_model, :n_heads] => ByRow(/) => :kv_size,
        [:d_model, :n_layers] => ByRow(/) => :aspect_ratio,
    )
    df = shuffle!(df) #[h, :]

    # Propose Dataset and LR Sweeps
    df = combine(groupby(df, :N)) do gdf
        n = first(gdf.N)
        D = plan_data_sweep(post_model, n)
        lr = plan_lr_sweep(post_model, n)
        df_sweep = allcombinations(DataFrame; D, lr)
        crossjoin(gdf, df_sweep)
    end

    # Truncate to 128
    df = shuffle!(df)[df.D .< 1e9, :]
    return df
end

function realize_plan(plan, nodes, batch_size; io=stdout, gas=64)

    for row in eachrow(plan)
        steps = round(row.D / (nodes * batch_size * gas); sigdigits=3)
        steps = Int(round(steps, RoundNearest))
        config = Dict(
            "nodes" => nodes,
            "train" => Dict(
                "trainer.max_steps" => steps,
                "trainer.accumulate_grad_batches" => gas,
                "model.lr_schedule.num_training_steps" => steps,
                "model.intermediate_size" => Int(row.d_ff),
                "model.hidden_size" => Int(row.d_model),
                "model.num_hidden_layers" => Int(row.n_layers),
                "model.num_attention_heads" => Int(row.n_heads),
                "model.optimizer.lr" => row.lr,
                "data.batch_size" => batch_size,
            )
        )
        println(io, "'$(JSON.json(config))'")
    end
end
