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
_scale_bins(scale::typeof(log10), bins::Int, x) = logrange(extrema(x)...; length=bins + 1)

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
    return ((l, u), (l, u))
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

function rescale(x)
    lb, ub = extrema(x)
    scale = inv(ub - lb)
    return @. (x - lb) * scale
end

function plot_chain_covariance(chains::ComponentArray{<:Real,3}; nbins=:scott)
    f = Figure(; size=96 .* (3.42, 3.42))
    params = ComponentArrays.labels(chains[1, 1, :])
    nparams = size(chains, 3)
    for (i, px) in enumerate(params)
        vx = vec(selectdim(chains, 3, i)) |> rescale
        vx_lims = extrema(vx)
        nbins_x = 3 * get_nbins(nbins, vx)
        for (j, py) in enumerate(params)
            vy = vec(selectdim(chains, 3, j)) |> rescale
            vy_lims = extrema(vy)
            ax = Axis(f[j, i]; ylabel=py, xlabel=px, limits=(vx_lims, vy_lims))
            nbins_y = 3 * get_nbins(nbins, vy)
            if i == j
                density!(ax, vx)
                ax.limits[] = (nothing, (0, nothing))
            elseif i > j
                hexbin!(ax, vx, vy;
                    bins=(nbins_x, nbins_y),
                    colorscale=Makie.Symlog10(1),
                    threshold=0,
                )
            else
                datashader!(ax, Point2f.(zip(vx, vy)))
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
    rowgap!(f.layout, 3)
    colgap!(f.layout, 3)
    resize_to_layout!(f)
    return f
end

""" Plot MCMC chains against each other to assess convergence """
function plot_chains(chains::AbstractChains)

    # Setup grid
    n_samples, n_chains, n_params = size(chains)
    f = Figure(; size=96 .* (4, n_params * 0.75))
    gl = GridLayout(f[1, 1], n_params, 1)

    # Plot Chains
    chain_axes = []
    labels = ComponentArrays.labels(chains[1, 1, :])
    for (i, label) in enumerate(labels)
        samples = selectdim(chains, 3, i)
        is_last = i == n_params
        ax = Axis(gl[i, 1];
            xlabel="Iteration",
            ylabel=string(label),
            limits=((0, n_samples), nothing),
            yticklabelsvisible=false,
            yticksvisible=false,
            ygridvisible=false,
            xgridvisible=true,
            xticksvisible=is_last,
            xticklabelsvisible=is_last,
            xlabelvisible=is_last,
        )
        push!(chain_axes, ax)
        ax_hist = Axis(gl[i, 2];
            limits=((0, nothing), nothing)
        )
        hidedecorations!(ax_hist)
        linkyaxes!(ax, ax_hist)
        for chain in eachslice(samples; dims=2)
            h = lines!(ax, chain; linewidth=1)
            density!(ax_hist, vec(chain),
                direction=:y,
                color=@lift(Makie.alphacolor($(h.color), 0.5 / n_chains)),
                linestyle=h.linestyle,
                strokecolor=h.color,
                strokewidth=1,
            )
        end
    end
    linkxaxes!(chain_axes...)
    colsize!(gl, 1, Relative(0.8))
    colgap!(gl, 3)
    rowgap!(gl, 5)
    resize_to_layout!(f)
    return f
end
