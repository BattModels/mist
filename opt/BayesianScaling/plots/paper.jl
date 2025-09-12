using Makie
using MISTStyle
using DataFrames
using StatsBase
using JLD2: jldopen
using BayesianScaling
using BayesianScaling: pf_day, get_nbins

function mark_model!(ax_scale, ax_compute;
    model_loss,
    d_model,
    ff_ratio,
    n_layers,
    steps,
    eff_batch_size,
    colorbar,
    h_loss,
    kwargs...
)
    N = BayesianScaling.non_embedding_size(d_model, ff_ratio * d_model, n_layers)
    D = steps * eff_batch_size
    C = 6 * float(N) * float(D)
    kwargs = (;
        marker=:star5,
        color=model_loss,
        colormap=colorbar.colormap,
        colorrange=colorbar.limits,
        colorscale=colorbar.scale,
        strokewidth=lift(x -> 1.5x, h_loss.strokewidth),
        strokecolor=:white,
        markersize=@lift(2 * $(h_loss.markersize)),
    )
    h_scale = scatter!(ax_scale, C ./ pf_day, N; kwargs...)
    h_compute = scatter!(ax_compute, C ./ pf_day, model_loss; kwargs...)

    return h_scale, h_compute
end

function mark_model!(ax_scale, ax_compute, text, text_pos; fontsize=6pt, kwargs...)
    h_scale, h_compute = mark_model!(ax_scale, ax_compute; kwargs...)
    annotation!(ax_compute, text_pos, lift(first, h_compute.positions)[];
        text,
        fontsize,
    )
    return h_scale, h_compute
end


scaling(x) = haskey(x[1,1,:], :scaling) ? selectdim(x, 3, :scaling) : x

function plot_covariance!(f, chains)
    # Extract data
    α = selectdim(chains, 3, :α)
    β = selectdim(chains, 3, :β)
    balance = @. α / β
    E = selectdim(chains, 3, :E)
    balance = vec(balance)
    E = vec(E)

    # Setup Plots
    ax = Axis(f[1, 1];
        xlabel=L"\alpha/\beta",
        ylabel=L"E",
        xscale=identity,
        yscale=log10,
    )
    h = hexbin!(ax, balance, E;
        colormap=:vik,
        threshold=10,
        bins=50,
    )
    colorrange = h.colorrange
    s = scatter!(ax, balance, E;
        color=1,
        alpha=1.0,
        marker=:circle,
        markersize=2pt,
        colorrange=(1, 10),
        colormap=h.colormap,
    )
    translate!(s, 0, 0, -1)
    return f
end

function plot_scaling_params!(f, chains; legend_pos=:top)
    A = selectdim(chains, 3, :A)
    B = selectdim(chains, 3, :B)
    α = selectdim(chains, 3, :α)
    β = selectdim(chains, 3, :β)
    E = selectdim(chains, 3, :E)
    G = @. ((α * A) / (β * B))^(1 / (α + β))
    ζ = @. (α * β) / (α + β)
    a = @. β / (α + β)
    b = @. α / (α + β)
    balance = @. α / β
    bins = get_nbins(:scott, a)

    hargs = (; bins, normalization=:probability, color=MISTStyle.UM_COLORS.blue)
    ax_args = (;
        ylabel=L"p(\cdot)",
        yticks=WilkinsonTicks(3),
    )

    prior_works = [
        "Kaplan et al." => (; α=0.076, β=0.103, a = 0.58, b = 0.42),
        "Hoffmann et al. (Appr. 1)" => (; a = 0.50, b = 0.50),
        "Hoffmann et al. (Appr. 3)" => (; α=0.34, β=0.28, a = 0.46, b = 0.54),
        "Bi et al. (Early)" => (; a = 0.450, b = 0.550),
        "Bi at al. (Current)" => (; a = 0.524, b = 0.478),
    ]

    gl = GridLayout(f[1, 1])

    vargs = (; linewidth=1pt, colormap=:tab10, colorrange=(1, 10))

   ax_a = Axis(gl[1, 1];
        limits=(nothing, (0, nothing)),
        xlabel=L"Convergance Rate: $\frac{\alpha \beta}{\alpha + \beta}$",
        ax_args...
    )
    hist!(ax_a, vec(ζ); hargs...)
    for (color, (label, s)) in enumerate(prior_works)
        haskey(s, :α) && haskey(s, :β) || continue
        vlines!(ax_a, (s.α * s.β) / (s.α + s.β); label, color, vargs...)
    end

    ax = Axis(gl[1, 2];
        limits=(nothing, (0, nothing)),
        xlabel=L"Model/Data Scaling: $\frac{\alpha}{\beta}$",
        ax_args...
    )
    hideydecorations!(ax, grid=false)
    linkyaxes!(ax_a, ax)
    hist!(ax, vec(balance); hargs...)
    for (color, (label, s)) in enumerate(prior_works)
        vlines!(ax, s.b / s.a; label, color, vargs...)
    end

    elem = map(enumerate(prior_works)) do (color, (label, _))
        LineElement(; color, label, vargs...)
    end
    halign = :center
    valign = :center
    if legend_pos == :top
        legend_pos = gl[0, 1:2]
        tellheight = true
        tellwidth = true
        orientation=:horizontal
        nbanks=2
    elseif legend_pos == :right
        legend_pos = gl[1, end+1]
        tellheight = false
        tellwidth = true
        orientation=:vertical
        nbanks=1
    elseif legend_pos == :left_inside
        legend_pos = gl[1, 1]
        tellheight = false
        tellwidth = false
        orientation=:vertical
        halign = :left
        valign = :top
        nbanks=1
    else
        error("Unknown legend position $legend_pos")
    end
    Legend(legend_pos, elem, MISTStyle.label.(elem);
        tellwidth,
        tellheight,
        margin=2pt .* (1, 1, 1, 1),
        orientation,
        nbanks,
        halign,
        valign,
    )

    return f
end

plot_prediction_intervals(model, chains; p=0.95) = plot_prediction_intervals!(Figure(), model, chains; p)
function plot_prediction_intervals!(f, model, chains; p=0.95)
    μ, lower, upper = BayesianScaling.predict_ci(model, chains; p)
    loss = response(model)
    @info "Prediction Interval R²" cor(loss, μ)

    ax = Axis(f[1, 1];
        xlabel="Validation Loss (nats)",
        ylabel="Expected Loss (nats)",
        xscale=log10,
        yscale=log10,
    )
    scatter!(ax, loss, μ;
        marker=:x,
        color=:blue
    )
    powerlaw!(ax, 1, 1;
        color=:black,
        linewidth=2pt,
        linestyle=:dash,
    )
    return f
end

function plot_lr_partial_dependence!(f, model, chains; p=0.95, npoints=100)
    lr_dev, residual = BayesianScaling.lr_partial_dependence(model.formula, chains, BayesianScaling.observations(model), response(model))
    c, cl, cu = BayesianScaling.credible_interval(chains[:, :, :lr][:,:, :penalty]; p)
    @info model.formula
    x_range = extrema(first, lr_dev)
    x_range = (0.9, 1.1) .* x_range
    local x, xsq
    if model.formula.harmonic_shape_penalty
        x = range(x_range..., length=npoints)
        xsq = x .^2
    else
        x = logrange(x_range..., length=npoints)
        xsq = @. log(x)^2
    end
    r_mu = c .* xsq
    r_lower = cl .* xsq
    r_upper = cu .* xsq
    if model.formula.geometric_penalty
        @. r_mu .= exp(r_mu)
        @. r_lower .= exp(r_lower)
        @. r_upper .= exp(r_upper)
    end

    ax = Axis(f[1, 1];
        limits=(extrema(x), nothing),
        xlabel=L"\eta/\eta_{\star}",
        ylabel=L"P_{\eta}",
        xscale=model.formula.harmonic_shape_penalty ? identity : log2,
        yscale=model.formula.geometric_penalty ? log2 : identity,
    )
    if model.formula.harmonic_shape_penalty
        ax.xticklabelrotation[] = pi/6
    end
    scatter!(ax, first.(lr_dev), first.(residual);
        marker=:x,
        color=MISTStyle.UM_COLORS.arboretum
    )
    predictionband!(ax, x, r_mu, r_lower, r_upper; color=:black)
    return f
end

function figure_bayesian(data;
    N=logrange(1e5, 3e9; length=50),
    C=logrange(1e12, 2*pf_day; length=50),
    p=0.95,
    n_samples=500,
)
    f = Figure(; size=(5.2inch, 3inch), figure_padding=(2pt, 2pt, 2pt, 4pt))

    model = data["model"]
    chains = data["chains"]
    chain_samples = BayesianScaling.subsample(chains, n_samples)
    df = DataFrame(model.observations)
    df.loss = model.response

    model_loss = StatsBase.response(model)
    model_size = df.model_size
    model_flops = @. 6 * float(df.model_size) * float(df.data_size)

    gl = GridLayout(f[1, 1])
    sublabel!(gl[1,1, TopLeft()], "a"; left=10)
    sublabel!(gl[1,2, TopLeft()], "b"; left=5)
    ax_compute = Axis(gl[1, 1];
        xlabel="Compute Budget (PF-Days)",
        limits=(extrema(C) ./ pf_day, (2e-3, 2)),
        xscale=log10,
        ylabel="Validation Loss (nats)",
        yscale=log10,
    )
    ax_scale = Axis(gl[1, 2];
        xscale=log10,
        yscale=log10,
        limits=(extrema(C) ./ pf_day, extrema(N)),
        xlabel="Compute Budget (PF-Days)",
        ylabel="Non-Embedding Parameters",
    )

    loss_c_opt = BayesianScaling.compute_optimal_loss(scaling(chain_samples), C; p)
    h_opt = predictionband!(ax_compute, C ./ pf_day, loss_c_opt...;
        color=MISTStyle.UM_COLORS.maize,
        linewidth=2pt,
        band_color=(RGBf(0.596, 0.612, 0.592), 0.4),
    )
    loss_cs = first(BayesianScaling.hoffman_compute_scaling(scaling(chain_samples), N, C))'
    levels = logrange(minimum(loss_cs), 1.0; length=10)
    cb = Colorbar(gl[1, end+1];
        label="Validation Loss (nats)",
        colormap=MISTStyle.CONTINUOUS_COLORS,
        limits=extrema(levels),
        scale=log10,
    )
    h = contourf!(ax_scale, C ./ pf_day, N, loss_cs;
        levels,
        colormap=cb.colormap,
        colorscale=cb.scale,
    )
    h_loss = scatter!(ax_compute, model_flops ./ pf_day, model_loss;
        marker=:circle,
        color=model_loss,
        colormap=cb.colormap,
        colorrange=cb.limits,
        colorscale=cb.scale,
        strokecolor=:black,
        strokewidth=0.1pt,
    )


    N_opt = BayesianScaling.compute_optimal_model_size(scaling(chain_samples), C)
    predictionband!(ax_scale, C ./ pf_day, N_opt...;
        color=h_opt.color,
        linewidth=h_opt.linewidth,
        band_color=h_opt.band_color,
    )
    h = scatter!(ax_scale, model_flops ./ pf_day, model_size;
        marker=h_loss.marker,
        color=model_loss,
        colormap=h_loss.colormap,
        colorrange=h_loss.colorrange,
        colorscale=h_loss.colorscale,
        strokewidth=h_loss.strokewidth,
        strokecolor=h_loss.strokecolor,
    )

    # 4yzwys2z
    mark_model!(ax_scale, ax_compute, "MIST-228M", (-15pt, -10pt);
        d_model=1024,
        ff_ratio=4,
        n_layers=18,
        steps=374_685,
        eff_batch_size=4*8*16*32,
        model_loss=0.01019080262631178,
        colorbar=cb,
        h_loss,
        fontsize=6pt,
    )

    # dh61satt
    mark_model!(ax_scale, ax_compute, "MIST-1.8B", (-5pt, 10pt);
        d_model=2304,
        ff_ratio=4,
        n_layers=28,
        steps=499_999,
        eff_batch_size=4*8*16*8,
        model_loss=0.03703538700938225,
        colorbar=cb,
        h_loss,
    )


    # Fit distributions
    gl = GridLayout(f[2,1])
    sublabel!(gl[1,1, TopLeft()], "c"; left=10)
    plot_scaling_params!(gl, scaling(chains))

    plot_covariance!(gl[1,2], scaling(chains))
    sublabel!(gl[1,2, TopLeft()], "d"; left=5)

    if model.formula isa BayesianScaling.ShapedScaling
        sublabel!(gl[1, 3, TopLeft()], "e"; left=12)
        plot_lr_partial_dependence!(gl[1, 3], model, chains)
        colsize!(gl, 3, Auto(0.5))
    end

    # Nudge layout
    colsize!(gl, 1, Auto(2))
    colgap!(gl, 3pt)
    rowgap!(f.layout, 6pt)
    resize_to_layout!(f)

    return f
end

function figure_bayesian_panel(data;
    N=logrange(1e5, 3e9; length=50),
    C=logrange(1e12, 2*pf_day; length=50),
    p=0.95,
    n_samples=500,
)
    f = Figure(; size=(89mm, 170mm), figure_padding=(2, 2, 2, 4))

    model = data["model"]
    chains = data["chains"]
    chain_samples = BayesianScaling.subsample(chains, n_samples)
    df = DataFrame(model.observations)
    df.loss = model.response

    model_loss = StatsBase.response(model)
    model_size = df.model_size
    model_flops = @. 6 * float(df.model_size) * float(df.data_size)

    # Plot Covariance
    colsize!(f.layout, 1, 100mm)
    rowsize!(f.layout, 1, 25mm)
    sublabel!(f[1, 1, TopLeft()], "a"; left=5)
    gl = GridLayout(f[2,1])
    plot_covariance!(gl[1,1], scaling(chains))
    plot_lr_partial_dependence!(gl[1, 2], model, chain_samples)
    sublabel!(gl[1, 1, TopLeft()], "b"; left=5)
    sublabel!(gl[1, 2, TopLeft()], "c"; left=5)

    # Scaling Laws
    gl = GridLayout(f[3, 1])
    sublabel!(gl[1,1, TopLeft()], "d"; left=5)
    sublabel!(gl[1,2, TopLeft()], "e"; left=5)
    ax_compute = Axis(gl[1, 1];
        xlabel="Compute Budget (PF-Days)",
        limits=(extrema(C) ./ pf_day, nothing),
        xscale=log10,
        ylabel="Validation Loss (nats)",
        yscale=log10,
    )
    ax_scale = Axis(gl[1, 2];
        xscale=log10,
        yscale=log10,
        limits=(extrema(C) ./ pf_day, extrema(N)),
        xlabel="Compute Budget (PF-Days)",
        ylabel="Non-Embedding Parameters",
    )

    loss_c_opt = BayesianScaling.compute_optimal_loss(scaling(chain_samples), C; p)
    h_opt = predictionband!(ax_compute, C ./ pf_day, loss_c_opt...;
        color=MISTStyle.UM_COLORS.maize,
        linewidth=2pt,
        band_color=(RGBf(0.596, 0.612, 0.592), 0.4),
    )
    loss_cs = first(BayesianScaling.hoffman_compute_scaling(scaling(chain_samples), N, C))'
    levels = logrange(minimum(loss_cs), 1.0; length=10)
    cb = Colorbar(gl[1, end+1];
        label="Validation Loss (nats)",
        colormap=MISTStyle.CONTINUOUS_COLORS,
        limits=extrema(levels),
        scale=log10,
    )
    h = contourf!(ax_scale, C ./ pf_day, N, loss_cs;
        levels,
        colormap=cb.colormap,
        colorscale=cb.scale,
    )
    h_loss = scatter!(ax_compute, model_flops ./ pf_day, model_loss;
        marker=:circle,
        color=model_loss,
        colormap=cb.colormap,
        colorrange=cb.limits,
        colorscale=cb.scale,
        strokecolor=:black,
        strokewidth=0.1pt,
    )


    N_opt = BayesianScaling.compute_optimal_model_size(scaling(chain_samples), C)
    predictionband!(ax_scale, C ./ pf_day, N_opt...;
        color=h_opt.color,
        linewidth=h_opt.linewidth,
        band_color=h_opt.band_color,
    )
    h = scatter!(ax_scale, model_flops ./ pf_day, model_size;
        marker=h_loss.marker,
        color=model_loss,
        colormap=h_loss.colormap,
        colorrange=h_loss.colorrange,
        colorscale=h_loss.colorscale,
        strokewidth=h_loss.strokewidth,
        strokecolor=h_loss.strokecolor,
    )

    # 4yzwys2z
    mark_model!(ax_scale, ax_compute, "MIST-228M", (-10pt, -10pt);
        d_model=1024,
        ff_ratio=4,
        n_layers=18,
        steps=374_685,
        eff_batch_size=4*8*16*32,
        model_loss=0.01019080262631178,
        colorbar=cb,
        h_loss,
    )

    # dh61satt
    mark_model!(ax_scale, ax_compute, "MIST-1.8B", (-5pt, 10pt);
        d_model=2304,
        ff_ratio=4,
        n_layers=28,
        steps=499_999,
        eff_batch_size=4*8*16*8,
        model_loss=0.03703538700938225,
        colorbar=cb,
        h_loss,
    )

    # Fit distributions
    plot_scaling_params!(f[4, 1], scaling(chains); legend_pos=:left_inside)
    sublabel!(f[4, 1, TopLeft()], "f"; left=5)

    # Nudge layout
    # colsize!(gl, 4, Relative(0.4))
    # colgap!(gl, 3pt)
    # colgap!(gl, 3, 6pt)
    rowgap!(f.layout, 6pt)
    resize_to_layout!(f)

    return f
end

function plot_model(model; kwargs...)
    @info "plotting $model"
    run_name = basename(model)
    data = jldopen(joinpath(model, "chains.jld2"), "r")
    try
        with_theme(MISTStyle.theme()) do
            figure_bayesian(data; kwargs...)
        end |> MISTStyle.savefig("bayesian-$run_name")

        with_theme(MISTStyle.theme()) do
            plot_prediction_intervals(data["model"], data["chains"])
        end |> MISTStyle.savefig("parity-$run_name")

    catch e
        @error "Failed to plot $model" e

    finally
        close(data)
    end

    return nothing
end

function plot_all(dir; kwargs...)
    for model in readdir(dir; join=true)
        isfile(joinpath(model, "chains.jld2")) || continue
        plot_model(model)
    end
end
