using Makie
using MISTStyle
using DataFrames
using StatsBase
using JLD2: jldopen
using BayesianScaling
using BayesianScaling: pf_day, get_nbins

function mark_model!(ax, d_model, ff_ratio, n_layers, steps, eff_batch_size; kwargs...)
    N = BayesianScaling.non_embedding_size(d_model, ff_ratio * d_model, n_layers)
    D = steps * eff_batch_size
    C = 6 * float(N) * float(D)
    return scatter!(ax, C ./ pf_day, N; kwargs...)
end

function figure_bayesian(data;
    N=logrange(1e5, 3e9; length=50),
    C=logrange(1e12, 2*pf_day; length=50),
    p=0.95,
)
    f = Figure(; size=(4.5inch, 3inch))

    model = data["model"]
    chains = BayesianScaling.subsample(data["chains"], 500)
    df = DataFrame(model.observations)
    df.loss = model.response
    sort!(df, :loss; rev=true)

    model_loss = StatsBase.response(model)
    model_size = df.model_size
    model_flops = @. 6 * float(df.model_size) * float(df.data_size)

    ax_compute = Axis(f[1, 1];
        xlabel="Compute Budget [PF-Days]",
        limits=(extrema(C) ./ pf_day, nothing),
        xscale=log10,
        ylabel="Validation Loss [nats]",
        yscale=log10,
    )
    sublabel!(f[1,1, TopLeft()], "a"; left=5)
    chains_scaling = haskey(chains[1, 1, :], :scaling) ? selectdim(chains, 3, :scaling) : chains
    loss_c_opt = BayesianScaling.compute_optimal_loss(chains_scaling, C; p)
    predictionband!(ax_compute, C ./ pf_day, loss_c_opt...;
        color=:red,
        band_color=(RGBf(0.596, 0.612, 0.592), 0.4),
    )
    scatter!(ax_compute, model_flops ./ pf_day, model_loss; marker=:circle)

    # Compute Optimal Frontier
    gl = GridLayout(f[1, 2])
    sublabel!(f[1,2, TopLeft()], "b"; left=5)
    ax_scale = Axis(gl[1, 1];
        xscale=log10,
        yscale=log10,
        limits=(extrema(C) ./ pf_day, extrema(N)),
        xlabel="Compute Budget [PF-Days]",
        ylabel="Model Size",
    )
    loss_cs = first(BayesianScaling.hoffman_compute_scaling(chains_scaling, N, C))'
    levels = logrange(minimum(loss_cs), 1.0; length=10)
    h = contourf!(ax_scale, C ./ pf_day, N, loss_cs;  levels, colorscale=log10)
    cb = Colorbar(gl[1, 2], h; label="Loss",)

    N_opt = BayesianScaling.compute_optimal_model_size(chains_scaling, C)
    predictionband!(ax_scale, C ./ pf_day, N_opt...;
        color=MISTStyle.UM_COLORS.maize,
        band_color=(RGBf(0.596, 0.612, 0.592), 0.4),
    )
    h = scatter!(ax_scale, model_flops ./ pf_day, model_size;
        marker=:circle,
        color=model_loss,
        colormap=cb.colormap,
        colorrange=cb.limits,
        colorscale=cb.scale,
        strokewidth=0.5,
        strokecolor=:black,
    )

    # 4yzwys2z
    mark_model!(ax_scale, 1024, 4, 18, 374685, 4*8*16*32;
        color=0.01019080262631178,
        marker=:star5,
        colormap=cb.colormap,
        colorrange=cb.limits,
        colorscale=cb.scale,
        strokewidth=h.strokewidth,
        strokecolor=h.strokecolor,
        markersize=@lift(2 * $(h.markersize)),
    )

    # dh61satt
    mark_model!(ax_scale, 2304, 4, 28, 2*499_999, 4*8*16*8;
        color=0.03703538700938225,
        marker=:star5,
        colormap=cb.colormap,
        colorrange=cb.limits,
        colorscale=cb.scale,
        strokewidth=h.strokewidth,
        strokecolor=h.strokecolor,
        markersize=@lift(2 * $(h.markersize)),
    )

    if !haskey(chains[1, 1, :], :scaling)
        resize_to_layout!(f)
        return f
    end

    # Fit distributions
    gl_fit = GridLayout(f[2, 1])
    sublabel!(f[2,1, TopLeft()], "c"; left=5)
    A = selectdim(chains_scaling, 3, :A)
    B = selectdim(chains_scaling, 3, :B)
    α = selectdim(chains_scaling, 3, :α)
    β = selectdim(chains_scaling, 3, :β)
    E = selectdim(chains_scaling, 3, :E)
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    a = @. β / (α + β)
    ax_a = Axis(gl_fit[1, 1];
        xlabel="a", ylabel=L"p(a)",
        limits=((0, 1), (0, nothing))
    )
    hidexdecorations!(ax_a)
    bins = get_nbins(:scott, a)
    hist!(ax_a, vec(a); bins, normalization=:pdf)
    cb = Colorbar(gl_fit[3, 1:3];
        colorrange=(1e-1, 1e2),
        scale=log10,
        vertical=false,
        flipaxis=false,
        label="Joint Density",
    )
    cb_kwargs = (; colorscale=cb.scale, colorrange=cb.colorrange, colormap=cb.colormap)

    ax_G = Axis(gl_fit[1, 2];
        xlabel="G", ylabel=L"p(G)",
        limits=(nothing, (0, nothing)),
        yticks=WilkinsonTicks(3),
        xscale=log10,
    )
    hidexdecorations!(ax_G)
    loghist!(ax_G, vec(G); bins, normalization=:pdf)

    ax_E = Axis(gl_fit[1, 3];
        xlabel="E", ylabel=L"p(E)",
        limits=(nothing, (0, nothing)),
        xscale=log10,
        yticks=WilkinsonTicks(3),
    )
    hidexdecorations!(ax_E)
    loghist!(ax_E, vec(E); bins, normalization=:pdf)

    ax = Axis(gl_fit[2, 1];
        limits=((0, 1), nothing),
        xlabel="a", ylabel="G",
        yscale=log10,
        yticks=LogTicks(WilkinsonTicks(3)),
    )
    hexbin!(ax, vec(a), vec(G); bins, cb_kwargs...)
    linkxaxes!(ax_a, ax)

    ax = Axis(gl_fit[2, 2];
        xlabel="G", ylabel="E",
        xscale=log10, yscale=log10,
    )
    hexbin!(ax, vec(G), vec(E); bins, cb_kwargs...)
    linkxaxes!(ax_G, ax)

    ax = Axis(gl_fit[2, 3];
        limits=(nothing, (0, 1)),
        xlabel="E", ylabel="a",
        xticks=LogTicks(WilkinsonTicks(5)),
        xscale=log10,
    )
    hexbin!(ax, vec(E), vec(a); bins, cb_kwargs...)

    # Learning Rate
    gl_penalty = GridLayout(f[2, 2])
    sublabel!(f[2,2, TopLeft()], "d"; left=5)

    if model.formula.lr_model_size == :d_model
        model_size = logrange(64, 4096; length=20)
        model_size_label = L"d_{model}"
    else
        model_size = N
        model_size_label = L"$$Model Size"
    end
    batch_size = logrange(32, 250_000; length=20)
    lr_opt = BayesianScaling.ideal_lr(model.formula, chains, model_size, batch_size; p)
    ax = Axis(gl_penalty[1, 1];
        xlabel="Batch Size",
        ylabel=model_size_label,
        xscale=log10, yscale=log10,
        limits=(extrema(batch_size), extrema(model_size)),
    )
    h = contourf!(ax, batch_size, model_size, first(lr_opt)';
        colorscale=log10,
        levels=logrange(extrema(first(lr_opt)')..., length=10),
    )
    Colorbar(gl_penalty[1, 2], h; label="Learning Rate")

    exp_model_size = map(run -> run[model.formula.lr_model_size], model.observations)
    exp_batch_size = map(run -> run.effective_batch_size, model.observations)
    θ_mle = BayesianScaling.maximum_posterior_estimate(model, chains)
    y = BayesianScaling.response(model)
    y_hat = BayesianScaling.predict(model, θ_mle)
    y_mape = @. abs(y - y_hat) / y
    h = scatter!(ax, exp_batch_size, exp_model_size;
        color=y_mape,
        colormap=:imola,
        colorscale=log10,
    )
    Colorbar(gl_penalty[2, 1:2], h;
        vertical=false,
        flipaxis=false,
        label="Residual Relative Error for Validation Loss",
    )


    resize_to_layout!(f)
    return f
end

function plot_all(dir; kwargs...)
    for model in readdir(dir; join=true)
        isfile(joinpath(model, "chains.jld2")) || continue
        @info "plotting $model"
        data = jldopen(joinpath(model, "chains.jld2"), "r")
        f = figure_bayesian(data; kwargs...)
        MISTStyle.savefig("bayesian", f; fig_dir=model)
    end
end

