"""
    predictionband(center, lower, upper)
using Makie: FigureLike

Plots a prediction band with median line overlayed
"""
@recipe(PredictionBand, x, center, lower, upper) do scene
    Theme(
        color=Makie.inherit(scene, (:Lines, :linecolor), :black),
        band_color=(:blue, 0.1),
    )
end
function Makie.plot!(plt::PredictionBand)
    band!(plt, plt.x, plt.lower, plt.upper; color=plt.band_color)
    lines!(plt, plt.x, plt.center; color=plt.color)
    return plt
end

@recipe(SampleRange, x, samples) do scene
    Theme(
        color=:black,
        markercolor=:blue,
        marker=:x,
        markersize=5,
        p=[0.025, 0.5, 0.975],
    )
end
function Makie.plot!(plt::SampleRange)
    l = stack(θ -> quantile(θ, plt.p[]), eachrow(plt.samples[]))
    satt = Makie.shared_attributes(plt, Scatter)
    satt.color = plt.markercolor
    rangebars!(plt, Makie.shared_attributes(plt, Makie.Rangebars), plt.x, l[1, :], l[3, :])
    scatter!(plt, satt, plt.x, l[2, :])
    return plt
end

"""
    loghist(x; scale, kwargs...)

Like `hist` but rescale bins to `logrange` if `scale` is `log10`
"""
@recipe(LogHist, x) do scene
    Theme(
        scale=log10,
        bins=Makie.inherit(scene, (:Hist, :bins), 15)
    )
end
function Makie.plot!(plt::LogHist)
    bins = lift(plt.x, plt.scale, plt.bins) do x, scale, bins
        _scale_bins(scale, bins, x)
    end
    attrs = Makie.shared_attributes(plt, Makie.Hist)
    attrs[:bins] = bins
    hist!(plt, plt.x; attrs...)
    return plt
end
_scale_bins(scale, bins, x) = bins
_scale_bins(scale::typeof(log10), bins::Int, x) = logrange(extrema(x)...; length=bins+1)

function plot_best_lr(chains::AbstractArray{<:Real,3}, df=missing;
    N=logrange(1e2, 1e10; length=200),
    lr=logrange(1e-5, 3e-3; length=100)
)
    f = Figure()

    # Prediction Plot
    # gl = GridLayout(f[1:2, 1])
    ax = Axis(f[1, 1];
        yscale=log10,
        xscale=log10,
        limits=(extrema(N), extrema(lr)),
        ylabel="Learning Rate, η",
        xlabel="Model Size (Non-Embedding)",
        ytickformat="{:.2e}",
    )
    lr_0 = vec(chains[:, :, :lr_0])
    lr_n = vec(chains[:, :, :lr_n])
    lr_p = vec(chains[:, :, :lr_p])

    # Plot LR from Kaplan, J. et al. 2020. Scaling Laws for Neural Language Models. arXiv.
    # LR(N) ≈ 0.003239 + −0.0001395 log(N )
    N_kaplan = logrange(768, 1.5e9; length=200)
    lr_kaplan = @. 0.003239 − 0.0001395 * log(N_kaplan)
    replace!(x -> x <= 0 ? NaN : x, lr_kaplan)
    lines!(ax, N_kaplan, lr_kaplan; color=:blue, label="Kaplan et al. 2020")

    # Add measured points
    if !ismissing(df)
        h = scatter!(ax, df.model_size, df.lr;
            color=df.min_val_loss,
            colorscale=log10,
            colorrange=extrema(df.min_val_loss),
            label="Empirical Data",
        )
        Colorbar(f[1, 2], h; label="Validation Loss", tickformat="{:.3f}")

    end
    scatter!(ax, non_embedding_size(768, 768, 12), 1.6e-4;
        marker=:star5, color=:red, markersize=15, label="MolFormer"
    )

    # Median Penalty
    lr_eff = @. exp(lr_0 - lr_n * log(N'))
    p = Matrix{Float32}(undef, length(lr), length(N))
    for i in 1:length(N)
        lr_penalty = @. lr_p * logsqdev(lr_eff[:, i], lr')
        for j in 1:length(lr)
            p[j, i] = quantile(lr_penalty[:, j], 0.95)
        end
    end
    h = contour!(ax, N, lr, p';
        levels=[0.1, 0.25, 0.5],
        labels=true,
        color=:red,
        label="95th-Percentile η Penalty",
    )

    # Median lr_eff
    q = map(c -> quantile(c, (0.5, 0.025, 0.975)), eachcol(lr_eff))
    predictionband!(ax, N, [l[1] for l in q], [l[2] for l in q], [l[3] for l in q];
        color=:black,
        band_color=(:slategray, 0.4),
        label="95% Credible Interval for η"
    )

    Legend(f[2, 1], ax; tellwidth=false, tellheight=true, nbanks=2)

    resize_to_layout!(f)
    return f
end

function plot_lamb_lr_scale(chains::ComponentArray{<:Real,3}, model;
    p=0.025,
    N=logrange(1e5, 1e10; length=100),
    batch_size=logrange(1e3, 1e6; length=100),
)

    f = Figure()
    ax = Axis(f[1, 1];
        xlabel="Effective Batch Size",
        ylabel="Model Size (Non-Embedding)",
        xscale=log10,
        yscale=log10,
        limits=(extrema(batch_size), extrema(N)),
    )
    N = collect(N)
    batch_size = collect(batch_size)
    a = vec(chains[:, :, :a])
    b = vec(chains[:, :, :b])
    c = vec(chains[:, :, :c])
    lamb_lr = Matrix{Float32}(undef, length(batch_size), length(N))
    uq = similar(lamb_lr)
    for (i, n) in enumerate(N)
        for (j, bs) in enumerate(batch_size)
            lr = @. exp(a + b * log(bs) + c * log(n))
            lamb_lr[j, i] = median(lr)
            uq[j, i] = std(lr) ./ lamb_lr[j, i]
        end
    end
    h = contourf!(ax, batch_size, N, lamb_lr;
        levels=10,
        colormap=:viridis,
        colorscale=log10,
    )
    contour!(ax, batch_size, N, uq; levels=5, color=:red, labels=true)
    Colorbar(f[1, 2], h; label="Learning Rate")

    lr_ideal = map(model.runs) do run
        lr = @. exp(a + b * log(run.effective_batch_size) + c * log(run.model_size))
        return median(lr)
    end
    ax = Axis(f[1, 3];
        xscale=log10, yscale=log10,
        limits=((1e-5, 1e-1), (1e-5, 1e-1)),
        aspect=1,
    )
    scatter!(ax, [run.lr for run in model.runs], lr_ideal;
        color=[run.loss for run in model.runs],
        colorscale=log10,
    )


    resize_to_layout!(f)
    return f
end


function plot_scaling(chains::AbstractArray{<:Real,3}, df;
    N=logrange(1e5, 1e10; length=75),
    C=logrange(1e10, 1e26; length=75)
)
    f = Figure()
    N = collect(N)
    C = collect(C)
    ax = Axis(f[1, 1];
        yscale=log10,
        xscale=log10,
        limits=(extrema(C) ./ pf_day, extrema(N)),
        xlabel="Compute Budget [PF-Days]",
        ylabel="Model Size",
    )


    loss_mean, loss_band = hoffman_compute_scaling(selectdim(chains, 3, :scaling), N, C)
    levels = logrange(1e-3, 2; length=20)
    h = contourf!(ax, C ./ pf_day, N, loss_mean'; levels, colorscale=log10)
    rel_band = reshape(diff(loss_band; dims=1), size(loss_mean)) ./ loss_mean
    contour!(ax, C ./ pf_day, N, rel_band';
        levels=10,
        labels=true,
        color=:red,
    )
    cb = Colorbar(f[1, 2], h; label="Loss")

    # Add Empirical Loss
    model_flops = @. 6 * float(df.model_size) * float(df.data_size) / pf_day
    sort!(df, [:loss, :model_size,]; rev=true)
    h = scatter!(ax, model_flops, df.model_size;
        color=df[:, :loss],
        colormap=cb.colormap,
        colorrange=cb.limits,
        colorscale=cb.scale,
        strokewidth=0.5,
        strokecolor=:black,
        marker=:x,
    )

    # Add Compute Optimal Frontier
    N_opt, N_band = compute_optimal_model_size(selectdim(chains, 3, :scaling), C)
    predictionband!(ax, C ./ pf_day, N_opt, selectdim(N_band, 1, 1), selectdim(N_band, 1, 2);
        color=:black,
        band_color=(:slategray, 0.4),
    )

    resize_to_layout!(f)
    return f
end

function plot_acquisition(chains, aq, df;
    N=logrange(1e5, 1e12; length=20),
    D=logrange(1e6, 10e15; length=20)
)
    acqustion = Matrix{Float32}(undef, length(D), length(N))
    N = collect(N)
    D = collect(D)
    for (i, d) in enumerate(D)
        for (j, n) in enumerate(N)
            acqustion[i, j] = aq(chains, n, d)
        end
    end
    f = Figure()
    ax = Axis(f[1, 1];
        xscale=log10,
        yscale=log10,
        limits=(extrema(D), extrema(N)),
        xlabel="Data Size [Obs.]",
        ylabel="Model Size (Non-Embedding)",
    )
    h = contourf!(ax, D, N, acqustion)
    scatter!(ax, df.data_size, df.model_size; color=df.min_val_loss, colorscale=log10)
    Colorbar(f[1, 2], h; label="Acquisition Function")

    resize_to_layout!(f)
    return f
end

dist_tf(::Any) = identity
dist_tf(::LogNormal) = log10

replace_domain_error(f) = Base.Fix1(replace_domain_error, f)
function replace_domain_error(f, x)
    try
        f(x)
    catch e
        if e isa DomainError
            return NaN
        else
            rethrow(e)
        end
    end
    return x
end


function plot_compute_optimal(chains::AbstractArray{<:Real,3}, df=missing;
    huber=nothing,
    p=0.95,
    C=logrange(1e8, 1e26; length=200)
)
    μ, band = compute_optimal_loss(chains, C; p)

    # Setup Plots
    f = Figure()
    gl = GridLayout(f[1, 1])
    ax = Axis(f[1, 1];
        ylabel="Estimated Compute-Optimal Loss",
        xlabel="Compute Budget (PF-Day)",
        limits=(extrema(C ./ pf_day), (1e-4, 1)),
        xscale=log10,
        yscale=log10,
    )
    predictionband!(ax, C ./ pf_day, μ, band[1, :], band[2, :])

    # Add Huber Fitted Model
    if !isnothing(huber)
        loss = map(x -> compute_optimal_loss(x; huber...), C)
        @info "huber" C loss
        lines!(ax, C ./ pf_day, loss; color=:red, label="Huber Fitted Model")
    end

    #  Add Training Data
    if !ismissing(df)
        scatter!(ax, @.(6 * float(df.data_size) * float(df.model_size) / pf_day), df.loss;
            marker=:x, color=:blue, label="Empirical",
        )
    end

    gl = GridLayout(f[1, 2])
    A = selectdim(chains, 3, :A)
    α = selectdim(chains, 3, :α)
    B = selectdim(chains, 3, :B)
    β = selectdim(chains, 3, :β)
    a = vec(@.(β / (α + β)))
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    bins = get_nbins(:scott, a)

    ax = Axis(gl[1, 1]; xlabel="a", limits=((0, 1), (0, nothing)))
    hideydecorations!(ax)
    hist!(ax, a; bins)

    ax = Axis(gl[2, 1]; xlabel="E", limits=(nothing, (0, nothing)), xscale=log10)
    hideydecorations!(ax)
    E = vec(selectdim(chains, 3, :E))
    loghist!(ax, E; bins, scale=ax.xscale)

    ax = Axis(gl[3, 1]; xlabel="G", limits=(nothing, (0, nothing)), xscale=log10)
    hideydecorations!(ax)
    loghist!(ax, vec(G); bins, scale=ax.xscale)
    colgap!(gl, 10)

    resize_to_layout!(f)
    return f
end

function parity_limits(x, y; margin=0.02)
    xl, xu = extrema(x)
    yl, yu = extrema(y)
    l = min(xl, yl) * (1 - margin)
    u = min(xu, yu) * (1 + margin)
    return ((l, u), (l,u))
end

function plot_penalty(model, chains::AbstractArray{<:Real,3}; p=0.95)

    # Plot Expected Loss vs. Measured Loss
    f = Figure()

    # Sample model
    chains = subsample(chains, 10)
    loss = StatsBase.response(model)
    loss_hat = expected_loss(model, chains)
    P = expected_penalties(model, chains)

    ax = Axis(f[1:2, 1];
        xlabel="Expected Loss",
        ylabel="Measured Loss",
        xscale=log10, yscale=log10,
        limits=parity_limits(loss, loss_hat),
    )

    loss_mu, loss_low, loss_high = credible_interval(loss_hat; p)
    rangebars!(ax, loss, loss_low, loss_high; color=:black, linewidth=0.5, direction=:x)
    scatter!(ax, loss_mu, loss; color=:red, marker=:x, markersize=3)

    # LR Penalty
    lr_ratios = logrange(1e-2, 1e2; length=50)
    ax = Axis(f[1, 2];
        xlabel="η/η₀",
        ylabel=L"y/\hat{y}",
        xscale=log2,
        yscale=log2,
    )
    lr = map(run -> run[:lr], model.runs)
    lr_ideal = ideal_lr(model, chains)
    lr_ideal = vec(mean(lr_ideal; dims=(1, 2)))
    lr_resid = penalty_residual(loss, loss_hat, P.lr)

    lr_sense = expected_lr_sensitivity(model, chains, lr_ratios)
    scatter!(ax, lr ./ lr_ideal, lr_resid; marker=:x, color=:red)
    predictionband!(ax, lr_ratios, credible_interval(lr_sense; p)...)

    ff = logrange(1, 16; base=2, length=100)
    ax = Axis(f[2, 2];
        xlabel="FF. Ratio",
        ylabel="y/E[L']",
    )
    df = DataFrame(model.runs)
    ff_resid = penalty_residual(loss, loss_hat, P.ff)

    ff_resid = penalty_residual(loss, loss_hat, P.ff)
    ff_sense = harmonic_penalty_posterior(selectdim(chains, 3, :ff_ratio), ff)
    predictionband!(ax, ff, credible_interval(ff_sense; p)...)
    scatter!(ax, df.ff_ratio, log.(ff_resid); color=:red)

    resize_to_layout!(f)
    return f
end


function plot_lr_map(m, chains::AbstractArray{<:Real,3}, df::DataFrame=DataFrame(m.runs);
    effective_batch_size=logrange(2^5, 2^25; length=20),
    model_size=logrange(1e5, 1e9; length=15),
    p=0.025
)
    lr_quantiles = ideal_lr_map(m, chains, model_size, effective_batch_size; p=0.95)
    lr_mean = selectdim(lr_quantiles, 3, 1)

    f = Figure()
    gl = GridLayout(f[1, 1])
    ax = Axis(gl[1, 1];
        xlabel="Effective Batch Size",
        xscale=log2,
        ylabel="Model Size",
        yscale=log10,
    )
    h_lr = contourf!(ax, effective_batch_size, model_size, lr_mean';
        colormap=:roma,
    )
    Colorbar(gl[1, 2], h_lr; scale=log10, label="Learning Rate")

    loss_hat = vec(mean(expected_loss(m, chains); dims=(1, 2)))
    loss = StatsBase.response(m)
    loss_residual = loss ./ loss_hat
    max_residula = maximum(abs, loss_residual)

    h_emp = scatter!(ax, df.effective_batch_size, df.model_size;
        color=loss_residual,
        colormap=:vik,
        colorrange=(-max_residula, max_residula),
        marker=:x,
        glowcolor=:black,
        glowwidth=3,
    )
    Colorbar(gl[2, 1], h_emp; label="Residual Loss", flipaxis=false, vertical=false)

    # Plot LR Scaling Distribution
    idx = ComponentArrays.label2index(chains[1, 1, :], "lr.ideal")
    ideal = selectdim(chains, 3, idx)
    gl = GridLayout(f[1, 2])

    for (idx, xlabel) in enumerate([L"\eta_0", "Eff. Batch Size", "Model Size"])
        ax = Axis(gl[idx, 1];
            xlabel,
            limits=(nothing, (0, nothing)),
            xscale=(idx == 1 ? log10 : identity),
        )
        x = vec(selectdim(ideal, 3, idx))
        loghist!(ax, x; bins=get_nbins(:scott, x), scale=ax.xscale)
        hideydecorations!(ax)
    end

    resize_to_layout!(f)
    return f
end

function _emperical_penalty(y::Vector, x::Vector, y_hat::Matrix, p, x0::AbstractVecOrMat)
    # Check shapes
    @assert length(y) == length(x) == size(y_hat, 1)
    @assert size(y_hat, 2) == length(p)
    if x0 isa AbstractVector
        @assert length(x0) == length(p)
        x0 = x0'
    else
        @assert size(x0) == size(y_hat)
    end

    d_mag = @. log(x) - log(x0)
    d = d_mag .^ 2
    penalty = vec(p)' .* d
    y_eff = y .- vec(median(y_hat .- penalty; dims=2))
    d_mag = vec(median(d_mag; dims=2))
    @assert y_eff isa Vector && d_mag isa Vector
    @assert length(y_eff) == length(y) == length(d_mag)
    return y_eff, exp.(d_mag)
end

""" Return indices for a nearly square grid of `n` plots """
function layout_indices(n::Int)
    nc = max(floor(Int, sqrt(n)), 1)
    nr = cld(n, nc)
    @assert nc * nr >= n
    return CartesianIndices((nr, nc))
end

function plot_residual_correlations(model, chains::AbstractArray{T,3}, df::DataFrame) where {T}
    y = mean(expected_loss(model, chains); dims=(1, 2)) |> vec
    y_hat = StatsBase.response(model)
    error = @. log(y) - log(y_hat)
    rescor = residual_correlations(error, df)

    # Sort by correlations
    cols = collect(keys(rescor))
    sort!(cols; by=x -> first(rescor[x]))

    f = Figure()
    ax = Axis(f[1, 1];
        xlabel="Residuals Quantiles",
        ylabel="Expected-Residual Quantiles",
        aspect=1,
    )
    σ = mean(selectdim(chains, 3, :sigma))
    qqplot!(ax, error, Normal(0, σ); qqline=:identity)


    # Plot Correlations
    gl = GridLayout(f[1, 2])
    indices = layout_indices(length(cols))
    for (c, idx) in zip(cols, indices)
        x = df[!, c]
        ax = Axis(gl[idx.I...]; xlabel=c, ylabel="Log-Residuals")
        hideydecorations!(ax)
        hidexdecorations!(ax)
        idx.I[2] != 1 && hideydecorations!(ax)
        scatter!(ax, x, error)
    end
    linkyaxes!(f.content...)
    colgap!(f.layout, 20)
    resize_to_layout!(f)

    return f
end

"""
    get_nbins(method::Symbol, x)
    get_nbins(nbins::Int, ::Any)

Estimate the number of bins for a histogram of `x` using the method `method`.
"""
get_nbins(method::Symbol, x) = get_nbins(Val(method), x)
get_nbins(::Val{:rice}, x) = max(ceil(Int, cbrt(2 * length(x))), 2)
get_nbins(nbins::Int, ::Any) = nbins
function get_nbins(::Val{:scott}, x)
    bw = 3.5 * std(x) / cbrt(length(x))
    l, u = extrema(x)
    nbins = ceil(Int, (u - l) / bw)
    return max(nbins, 2)
end

function plot_chain_covariance(chains::ComponentArray{<:Real,3}; nbins=:scott)
    f = Figure()
    params = ComponentArrays.labels(chains[1, 1, :])
    nparams = size(chains, 3)
    nsamples = prod(size(chains)[1:2])
    for (i, px) in enumerate(params)
        vx = vec(selectdim(chains, 3, i))
        nbins_x = get_nbins(nbins, vx)
        for (j, py) in enumerate(params)
            ax = Axis(f[j, i]; ylabel=py, xlabel=px)
            vy = vec(selectdim(chains, 3, j))
            nbins_y = get_nbins(nbins, vy)
            if i == j
                hist!(ax, vx; bins=nbins_x)
            else
                hexbin!(ax, vx, vy; bins=(nbins_x, nbins_y))
            end

            # Just show the labels
            hideydecorations!(ax)
            hidexdecorations!(ax)
            if i == 1
                ax.ylabelvisible = true
            end
            if j == nparams
                ax.xlabelvisible = true
            end
        end
    end
    rowgap!(f.layout, 5)
    colgap!(f.layout, 5)
    resize_to_layout!(f)
    return f
end

function plot_chains(chains::AbstractArray{<:Real,3}; nbins=:scott)
    f = Figure()
    params = ComponentArrays.labels(chains[1, 1, :])
    nparams = size(chains, 3)
    nsamples = size(chains, 1)

    # Setup grid
    if nparams <= 5
        n_row = nparams
        n_col = 1
    else
        n_col = div(nparams, 6)
        n_row = ceil(Int, nparams / n_col)
    end
    gl = GridLayout(f[1, 1], n_row, n_col)
    indices = CartesianIndices((n_row, n_col))

    # Plot Chains
    gl = GridLayout(f[1, 1])
    chain_axes = []
    for (i, p) in enumerate(params)
        samples = selectdim(chains, 3, i)
        fchain = GridLayout(gl[indices[i].I...])
        ax = Axis(fchain[1, 1];
            xlabel="step",
            ylabel=p,
            limits=((0, nsamples), nothing),
            yticklabelsvisible = false,
            yticksvisible = false,
        )
        push!(chain_axes, ax)
        ax_hist = Axis(fchain[1, 2];
            limits=((0, nothing), nothing)
        )
        hidedecorations!(ax_hist)
        if indices[i].I[1] != n_row
            hidexdecorations!(ax)
        end
        colgap!(fchain, 5)
        colsize!(fchain, 2, Relative(0.2))

        linkyaxes!(ax, ax_hist)
        for chain in eachslice(samples; dims=2)
            h = lines!(ax, chain; linewidth=1)
            hist!(ax_hist, vec(chain),
                bins=get_nbins(nbins, chain),
                normalization=:pdf,
                direction=:x,
                color=h.color,
            )
        end
    end
    linkxaxes!(chain_axes...)
    colgap!(gl, 5)
    rowgap!(gl, 5)
    resize_to_layout!(f)
    return f
end

function plot_training_progress(model, chains)
    f = Figure()
    del = 1e-4
    ax = Axis(f[1, 1];
        xlabel="Relative Training Progress",
        ylabel="Validation Loss",
        limits=((0, 1), (1e-3, 1.0)),
        xscale=identity,
        yscale=log10,
    )

    out = reduce((s...) -> cat(s...; dims=3), generated_quantities(model, chains))
    @assert size(out, 2) == 2
    mu = view(out, :, 1, :)
    mu_hoffman = view(out, :, 2, :)
    sl = Slider(f[1, 2], range=1:size(mu, 1), horizontal=false, tellwidth=true,)
    loss_trace = lift(sl.value) do idx
        loss = model.args.loss[idx]
        step = model.args.step[idx]
        Point2.(step, loss)
    end
    expected_loss = lift(sl.value) do idx
        step = range(0, 1; length=100)
        a = vec(chains[:, Symbol("loss_trace_a[$idx]"), :])
        b = vec(chains[:, Symbol("loss_trace_b[$idx]"), :])
        mu_step = median(step_log_loss.(step', mu[idx, :], a, b); dims=1) |> vec .|> exp
        Point2.(step, mu_step)
    end
    best_loss = lift(sl.value) do idx
        step = range(0, 1; length=100)
        a = vec(chains[:, Symbol("loss_trace_a[$idx]"), :])
        b = vec(chains[:, Symbol("loss_trace_b[$idx]"), :])
        mu_step = median(step_log_loss.(step', mu_hoffman[idx, :], a, b); dims=1) |> vec .|> exp
        Point2.(step, mu_step)
    end
    lines!(ax, loss_trace; color=:red)
    lines!(ax, expected_loss; color=:blue)
    lines!(ax, best_loss; color=:green)
    resize_to_layout!(f)
    return f
end

