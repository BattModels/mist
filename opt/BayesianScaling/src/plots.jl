"""
    predictionband(center, lower, upper)

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

function plot_best_lr(chains, df=missing;
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
    lr_0 = vec(chains[:, :lr_0, :])
    lr_n = vec(chains[:, :lr_n, :])
    lr_p = vec(chains[:, :lr_p, :])

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
            label="Emperical Data",
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

    return f
end

function plot_scaling(chains, df;
    N=logrange(1e1, 1e9; length=200),
    C=logrange(1e4, 1e26; length=200)
)
    f = Figure()
    N = collect(N)
    C = collect(C)
    ax = Axis(f[1, 1];
        yscale=log10,
        xscale=log10,
        limits=(extrema(C) ./ pf_day, extrema(N)),
        xlabel="Compute Budget (PF-Days)",
        ylabel="Model size",
    )

    loss = Matrix{Float32}(undef, length(C), length(N))
    loss_std = similar(loss)
    A = vec(chains[:, :A, :])
    α = vec(chains[:, :α, :])
    B = vec(chains[:, :B, :])
    β = vec(chains[:, :β, :])
    E = vec(chains[:, :E, :])
    for (i, n) in enumerate(N)
        for (j, c) in enumerate(C)
            d = c / (6n)
            l = @.((A / n^α) + (B / d^β) + E)
            loss[j, i] = median(l)
            loss_std[j, i] = std(log10.(l))
        end
    end
    loss_low = min(minimum(df.min_val_loss), minimum(loss))
    loss_high = maximum(df.min_val_loss)
    levels = logrange(loss_low, loss_high; length=20)
    h = contourf!(ax, C ./ pf_day, N, loss; levels, colorscale=log10)
    contour!(ax, C ./ pf_day, N, loss_std;
        levels=10,
        # levels=logrange(1e-3, 10; length=10),
        # levels=logrange(extrema(loss_std)...; length=10),
        labels=true,
        color=:red,
    )
    cb = Colorbar(f[1, 2], h; label="Validation Loss")

    # Add Empirical Loss
    model_flops = @. 6 * float(df.model_size) * float(df.data_size) / pf_day
    h = scatter!(ax, model_flops, df.model_size;
        color=df[:, :min_val_loss],
        colormap=cb.colormap,
        colorrange=cb.limits,
        colorscale=cb.scale,
        strokewidth=0.5,
        strokecolor=:black,
        label="Emperical Data"
    )
    Colorbar(f[1, 2], h; label="Validation Loss")

    # Add Compute Optimal Frontier
    n_opt = Matrix{Float32}(undef, 3, length(C))
    for (i, c) in enumerate(C)
        n_opt[:, i] .= quantile(compute_optimal_model(c; A, α, B, β, E), (0.025, 0.5, 0.975))
    end
    predictionband!(ax, C ./ pf_day, n_opt[2, :], n_opt[1, :], n_opt[3, :];
        color=:black,
        band_color=(:slategray, 0.4),
        label="Compute Optimal Frontier",
    )
    return f
end

function plot_parity(model, chains; p=0.025)
    y = sample_response(model, chains)
    yq = slice_quantile(y, (p, 0.5, 1-p); dims=1)

    f = Figure()
    ax = Axis(f[1, 1];
        yscale=log10,
        xscale=log10,
        ylabel="Predicted Loss",
        xlabel="Measured Loss",
    )
    # scatter!(ax, y, model.args.loss)
    y_loss = map(minimum, model.args.loss)
    rangebars!(ax, y_loss, yq[1, :], yq[3, :]; color=:blue, linewidth=1, label="95% Credible Interval")

    # Plot Parity line
    n = 100
    ll = minimum(yq)
    uu = maximum(yq)
    lines!(ax, range(ll, uu; length=n), range(ll, uu; length=n), color=:red)
    axislegend(ax, position=:rb)
    return f
end

function plot_scaling_parameters(chains, l=0, u=1)
    β = vec(chains[:β])
    α = vec(chains[:α])
    a = @.(β / (α + β)) |> vec
    b = @.(α / (α + β)) |> vec
    bins = floor(Int, sqrt(length(α)))
    normalization = :pdf
    f = Figure(size=(600, 600))
    ax_a = Axis(f[1, 1]; limits=((l, u), (0, nothing)))
    hidedecorations!(ax_a)
    hist!(ax_a, a; bins, normalization)
    ax_b = Axis(f[2, 2]; limits=((0, nothing), (l, u)))
    hidedecorations!(ax_b)
    hist!(ax_b, b; bins, normalization, direction=:x)

    ax = Axis(f[2, 1]; xlabel="a", ylabel="b",
        limits=((l, u), (l, u)),
        aspect=1,
    )
    h = hexbin!(ax, a, b; bins=bins)
    Colorbar(f[3, 1:2], h;
        vertical=false, tellheight=false, flipaxis=false,
        label="Count of Samples",
    )

    scatter!(ax, 0.46, 0.54; marker=:x, color=:red, label="Hoffmann et. al.")
    scatter!(ax, 0.73, 0.27; marker=:+, color=:blue, label="Kaplan et. al.")
    # Legend(f[1, 2], ax; tellwidth=false, tellheight=true)
    axislegend(ax, position=:lb)

    rowgap!(f.layout, 5)
    colgap!(f.layout, 5)
    colsize!(f.layout, 1, 300)
    colsize!(f.layout, 2, 50)
    rowsize!(f.layout, 1, 50)
    rowsize!(f.layout, 2, 300)
    rowsize!(f.layout, 3, 20)
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

function plot_chains(model, chains)
    f = Figure()
    ns, _, nc = size(chains)
    priors = Turing.DynamicPPL.extract_priors(model)
    names = chains.name_map.parameters
    np = length(names)

    # Setup grid
    if np <= 5
        n_row = np
        n_col = 1
    else
        n_col = div(np, 6)
        n_row = ceil(Int, np / n_col)
    end
    gl = GridLayout(f[1, 1], n_row, n_col)
    indices = CartesianIndices((n_row, n_col))

    # Plot Chains
    gl = GridLayout(f[1, 1])
    for (i, p) in enumerate(names)
        yscale = dist_tf(priors[Turing.@varname($p)])
        samples = replace_domain_error.(yscale, Float32.(chains[p]))
        ylims = extrema(samples)
        fchain = GridLayout(gl[indices[i].I...])
        ax = Axis(fchain[1, 1];
            xlabel="step",
            ylabel=string(p),
            yscale,
            limits=((1, ns), nothing)
        )
        ax_hist = Axis(fchain[1, 2];
            limits=((0, nothing), nothing)
        )
        hidedecorations!(ax_hist)
        if indices[i].I[1] != n_row
            hidexdecorations!(ax)
        end
        colgap!(fchain, 5)
        colsize!(fchain, 2, Relative(0.2))
        #
        # linkyaxes!(ax, ax_hist)
        for j in 1:nc
            h = lines!(ax, samples[:, j]; linewidth=1)
            hist!(ax_hist, vec(yscale.(samples));
                normalization=:pdf,
                direction=:x,
                color=h.color,
            )
        end
    end
    colgap!(gl, 5)
    rowgap!(gl, 5)

    ax = Axis(f[2, 1]; xlabel="step", ylabel="Auto-Correlation")
    lags = 1:fld(ns, 2)
    for pdx in 1:np
        ac = zeros(eltype(chains.value), length(lags), nc)
        for j in 1:nc
            ac[:, j] .= autocor(Array(chains[:, pdx, j]), lags)
        end
        mu = mean(ac; dims=2) |> vec
        s = std(ac; dims=2) |> vec
        predictionband!(ax, lags, mu, mu .- s, mu .+ s;
            label=names[pdx],
            band_color=(:slategray, 0.1),
        )
    end
    rowsize!(f.layout, 2, Relative(0.3))
    return f
end

function plot_compute_optimal(chains, df=missing;
    N=logrange(1e3, 1e12; length=200),
    C=logrange(1e8, 1e26; length=20)
)

    n_opt = Matrix{Float32}(undef, length(N), length(C))
    loss_quantiles = Matrix{Float32}(undef, 3, length(C))
    σ² = chains[:, :σ², :]
    for (i, c) in enumerate(C)
        n = compute_optimal_model(c;
            A=chains[:, :A, :],
            α=chains[:, :α, :],
            B=chains[:, :B, :],
            β=chains[:, :β, :],
        )

        # Estimate Loss of Compute-Optimal Model
        d = @. c / (6n)
        loss = hoffman_scaling(n, d;
            A=chains[:, :A, :],
            α=chains[:, :α, :],
            B=chains[:, :B, :],
            β=chains[:, :β, :],
            E=chains[:, :E, :],
        )
        @assert all(>(0), loss)
        for idx in eachindex(loss)
            loss[idx] = rand(LogNormal(log(loss[idx]), σ²[idx]))
        end
        loss_quantiles[:, i] .= quantile(vec(loss), (0.025, 0.5, 0.975))

        # Compute CDF of the compute-optimal model
        n_cdf = ecdf(vec(n))
        for (j, n) in enumerate(N)
            n_opt[j, i] = n_cdf(n)
        end
    end

    # Setup Plots
    f = Figure()
    C = collect(C) ./ pf_day
    N = collect(N)
    ax = Axis(f[1, 1];
        ylabel="Estimated Compute-Optimal Loss",
        xlabel="Compute Budget (PF-Day)",
        limits=(extrema(C), (1e-2, 1)),
        xscale=log10,
        yscale=log10,
        ytickformat="{:.3f}",
    )
    ax_cdf = Axis(f[1, 2];
        xlabel="Model Size (Non-Embedding)",
        ylabel="Emperical Cumulative Distribution Function",
        xscale=log10,
        limits=(extrema(N), (0, 1)),
    )
    cb = Colorbar(f[1, 3];
        label="Compute Budget (PF-Day)",
        colormap=:viridis,
        colorrange=extrema(C),
        scale=log10,
    )
    predictionband!(ax, C, loss_quantiles[2, :], loss_quantiles[1, :], loss_quantiles[3, :])
    for (i, c) in enumerate(C)
        lines!(ax_cdf, N, n_opt[:, i];
            color=c,
            colorscale=cb.scale,
            colormap=cb.colormap,
            colorrange=cb.colorrange
        )
    end

    if !ismissing(df)
        scatter!(ax, @.(6 * float(df.data_size) * float(df.model_size) / pf_day), df.min_val_loss;
            marker=:x, markersize=3, color=:blue, label="Emperical",
        )
    end

    return f
end

function plot_penalty(model, chains, deviance=logrange(1e-2, 1e2; length=100))

    f = Figure()
    ax_lr = Axis(f[1, 1]; xlabel="η/η_eff", ylabel="Penalty", xscale=log10, aspect=1)
    ax_ff = Axis(f[1, 2]; xlabel="Feed Forward Ratio", xscale=log2, aspect=1)
    ax_aspect = Axis(f[1, 3]; xlabel="Aspect Ratio", xscale=log2, aspect=1)
    hideydecorations!(ax_ff; grid=false)
    hideydecorations!(ax_aspect; grid=false)
    linkyaxes!(ax_lr, ax_ff, ax_aspect)

    # Setup
    deviance = collect(deviance)
    y = model.args.loss
    y_hat = stack(generated_quantities(model, chains))
    y_hat = reshape(y_hat, size(y_hat, 1), :)

    scatter_args = (;
        markersize=5,
        marker=:x,
        color=:blue,
    )

    # Plot LR Penalty and Measured Penalty
    lr_0 = vec(chains[:, :lr_0, :])
    lr_n = vec(chains[:, :lr_n, :])
    lr_p = vec(chains[:, :lr_p, :])
    p_lr = median(lr_p) .* log.(deviance) .^ 2
    lr_eff = @. lr_0' - lr_n' * log(model.args.model_size)
    lr_eff, lr_dev = _emperical_penalty(y, model.args.lr, y_hat, lr_p, exp.(lr_eff))
    scatter!(ax_lr, lr_dev, lr_eff; scatter_args...)
    penaltyband!(ax_lr, lr_p, ones(length(lr_p)), deviance; color=:red)

    # FF Ratio
    ff_ratio_0 = vec(chains[:, :ff_ratio_0, :])
    ff_ratio_p = vec(chains[:, :ff_ratio_p, :])
    ff_eff, _ = _emperical_penalty(y, model.args.ff_ratio, y_hat, ff_ratio_p, ff_ratio_0)
    scatter!(ax_ff, model.args.ff_ratio, ff_eff; scatter_args...)
    x = collect(logrange(extrema(model.args.ff_ratio)...; length=length(deviance)))
    penaltyband!(ax_ff, ff_ratio_p, ff_ratio_0, x; color=:red)

    # Aspect Ratio
    aspect_ratio_0 = vec(chains[:, :aspect_ratio_0, :])
    aspect_ratio_p = vec(chains[:, :aspect_ratio_p, :])
    aspect_eff, _ = _emperical_penalty(y, model.args.aspect_ratio, y_hat, aspect_ratio_p, aspect_ratio_0)
    scatter!(ax_aspect, model.args.aspect_ratio, aspect_eff; scatter_args...)
    x = collect(logrange(extrema(model.args.aspect_ratio)...; length=length(deviance)))
    penaltyband!(ax_aspect, aspect_ratio_p, aspect_ratio_0, x; color=:red)

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

function penaltyband!(ax, p, x0, x; kwargs...)
    p = @. p' * logsqdev(x, x0')
    q = col_quantile(p', (0.025, 0.5, 0.975))
    predictionband!(ax, x, q[2, :], q[1, :], q[3, :]; kwargs...)
end

""" Return indices for a nearly square grid of `n` plots """
function layout_indices(n::Int)
    nc = max(floor(Int, sqrt(n)), 1)
    nr = cld(n, nc)
    @assert nc * nr >= n
    return CartesianIndices((nr, nc))
end

function plot_residual_correlations(model, chains, df; p=0.25)
    y = first(model.args)
    y_hat = sample_response(model, chains)
    error = @. log(y) - log(y_hat)
    rescor = residual_correlations(error, df)

    # Sort by correlations
    cols = collect(keys(rescor))
    sort!(cols; by=x -> first(rescor[x]))

    f = Figure()
    error_q = col_quantile(error', (p, 1 - p))
    indices = layout_indices(length(cols))
    for (c, idx) in zip(cols, indices)
        x = df[!, c]
        x = x .* (1 .+ 0.01 * randn(length(x)))
        # if c == "min_val_loss" || (0 < minimum(x) && 2 <= -(-(extrema(log10.(x))...)))
        xscale = log10
        # else
        #     xscale = identity
        # end

        ax = Axis(f[idx.I...];
            xlabel=c,
            ylabel="Log-Residuals",
            xscale, e
        )
        idx.I[2] != 1 && hideydecorations!(ax)
        rangebars!(ax, x, error_q[1, :], error_q[2, :])
    end
    linkyaxes!(f.content...)
    colgap!(f.layout, 20)

    return f
end

function plot_chain_covariance(model, chains; skip_vector_parameters=true)
    f = Figure()
    names = deepcopy(chains.name_map[:parameters])
    if skip_vector_parameters
        # Skip parameters that end with [xx] (i.e. `loss_trace_a[10]`)
        filter!(names) do name
            match(r"\[\d*?\]$", string(name)) == nothing
        end
    end
    priors = Turing.DynamicPPL.extract_priors(model)
    nsamples = size(chains, 1) * size(chains, 3)
    bins = ceil(Int, sqrt(nsamples))
    for (i, px) in enumerate(names)
        xscale = dist_tf(priors[Turing.@varname($px)])
        for (j, py) in enumerate(names)
            yscale = dist_tf(priors[Turing.@varname($py)])
            ax = Axis(f[j, i];
                ylabel=string(py),
                xlabel=string(px),
            )
            if i == j
                hist!(ax, vec(chains[px]); bins)
            else
                hexbin!(ax, vec(chains[px]), vec(chains[py]); bins)
            end
            i != 1 && hideydecorations!(ax)
            j != length(names) && hidexdecorations!(ax)
        end
    end
    rowgap!(f.layout, 5)
    colgap!(f.layout, 5)
    return f
end

function plot_training_progress(model, chains::Chains)
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
    return f
end
