using Makie
using MISTStyle
using DataFrames
using StatsBase
using JLD2: jldopen
using BayesianScaling
using BayesianScaling: pf_day

function mark_model!(ax_scale, ax_compute; model_label=missing, model_loss, d_model, ff_ratio, n_layers, steps, eff_batch_size, colorbar, h_loss, kwargs...)
    N = BayesianScaling.non_embedding_size(d_model, ff_ratio * d_model, n_layers)
    D = steps * eff_batch_size
    C = 6 * float(N) * float(D)
    kwargs = (;
        marker=:star5,
        color=model_loss,
        colormap=colorbar.colormap,
        colorrange=colorbar.limits,
        colorscale=colorbar.scale,
        strokewidth=h_loss.strokewidth,
        strokecolor=h_loss.strokecolor,
        markersize=@lift(2 * $(h_loss.markersize)),
    )
    h_scale = scatter!(ax_scale, C / pf_day, N; kwargs...)
    h_compute = scatter!(ax_compute, C / pf_day, model_loss; kwargs...)
    if !ismissing(model_label)
        text, offset = model_label
        text!(ax_compute, Point2(C/ pf_day, model_loss); text, offset, align=(:center, :bottom))
    end
    return h_scale, h_compute
end

scaling(x) = haskey(x[1,1,:], :scaling) ? selectdim(x, 3, :scaling) : x

function plot_scaling_params!(f, chains)
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
    # bins = get_nbins(:scott, a)

    hargs = (; normalization=:probability, color=MISTStyle.UM_COLORS.blue)
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

    vargs = (; linewidth=1.5pt, colormap=:tab10, colorrange=(1, 10))
    elem = map(enumerate(prior_works)) do (color, (label, _))
        LineElement(; color, label, vargs...)
    end
    Legend(gl[2, 1:2], elem, MISTStyle.label.(elem);
        tellwidth=true,
        tellheight=true,
        nbanks=3,
        halign=:center,
        margin=2pt .* (1, 1, 1, 1),
        orientation=:horizontal,
    )

   ax_a = Axis(gl[1, 1];
        limits=(nothing, (0, nothing)),
        xlabel=L"Convergence Rate: $\frac{\alpha \beta}{\alpha + \beta}$",
        ax_args...
    )
    for (color, (label, s)) in enumerate(prior_works)
        haskey(s, :α) && haskey(s, :β) || continue
        vlines!(ax_a, (s.α * s.β) / (s.α + s.β); label, color, vargs...)
    end
    hist!(ax_a, vec(ζ); hargs...)


    ax = Axis(gl[1, 2];
        limits=(nothing, (0, nothing)),
        xlabel=L"Data/Model Scaling: $\frac{\alpha}{\beta}$",
        ax_args...
    )
    hideydecorations!(ax, grid=false)
    linkyaxes!(ax_a, ax)
    for (color, (label, s)) in enumerate(prior_works)
        vlines!(ax, s.b / s.a; label, color, vargs...)
    end
    hist!(ax, vec(balance); hargs...)

    ax = Axis(f[1, 2];
        xlabel=L"G",
        ylabel=L"E",
        xscale=log10,
        yscale=log10,
    )
    E = vec(E)
    G = vec(G)
    h = hexbin!(ax, G, E; colormap=:vik, threshold=5, bins=50)
    s = scatter!(ax, G, E;
        color=0,
        colormap=:vik,
        marker=:circle,
        markersize=2pt,
        colorrange=h.colorrange
    )
    translate!(s, 0, 0, -1)
    return f
end

function plot_prediction_intervals!(f, model, chains; p=0.95)
    μ, lower, upper = BayesianScaling.predict_ci(model, chains; p)
    loss = response(model)
    @info cor(loss, μ)

    ax = Axis(f[1, 1];
        xlabel="Validation Loss [nats]",
        ylabel="Expected Loss [nats]",
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
    scatter!(ax, first.(lr_dev), first.(residual); marker=:x, color=MISTStyle.UM_COLORS.arboretum)
    predictionband!(ax, x, r_mu, r_lower, r_upper; color=:black)
    return f
end

function figure_bayesian(data;
    N=logrange(1e5, 3e9; length=50),
    C=logrange(1e12, 2*pf_day; length=50),
    p=0.95,
    n_samples=500,
)
    f = Figure(; size=(680, 168))

    model = data["model"]
    chains = data["chains"]
    chain_samples = BayesianScaling.subsample(chains, n_samples)
    df = DataFrame(model.observations)
    df.loss = model.response

    model_loss = StatsBase.response(model)
    model_size = df.model_size
    model_flops = @. 6 * float(df.model_size) * float(df.data_size)

    gl = GridLayout(f[1, 1])
    # sublabel!(gl[1,1, TopLeft()], "a"; left=5)
    # sublabel!(gl[1,2, TopLeft()], "b"; left=5)
    ax_compute = Axis(gl[1, 1];
        xlabel="Compute Budget [PF-Days]",
        limits=(extrema(C) ./ pf_day, nothing),
        xscale=log10,
        ylabel="Validation Loss [nats]",
        yscale=log10,
    )
    ax_scale = Axis(gl[1, 2];
        xscale=log10,
        yscale=log10,
        limits=(extrema(C) ./ pf_day, extrema(N)),
        xlabel="Compute Budget [PF-Days]",
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
        label="Loss",
        colormap=MISTStyle.CONTINUOUS_COLORS,
        limits=extrema(levels),
        scale=log10,
    )
    h = contourf!(ax_scale, C ./ pf_day, N, loss_cs;
        levels,
        colormap=cb.colormap,
        # colorrange=cb.limits,
        colorscale=cb.scale,
    )
    h_loss = scatter!(ax_compute, model_flops ./ pf_day, model_loss;
        marker=:circle,
        color=model_loss,
        colormap=cb.colormap,
        colorrange=cb.limits,
        colorscale=cb.scale,
        strokecolor=:black,
        strokewidth=0.5,
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
    mark_model!(ax_scale, ax_compute;
        model_label="MIST-228M" => (0, 4),
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
    mark_model!(ax_scale, ax_compute;
        model_label="MIST-1.8B" => (-5, 5),
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
    gl = GridLayout(f[1,2])
    # sublabel!(gl[1,1, TopLeft()], "c"; left=5)
    plot_scaling_params!(gl, scaling(chains))
    # sublabel!(gl[1, 2, TopLeft()], "d"; left=5)

    if model.formula isa BayesianScaling.ShapedScaling
        # sublabel!(gl[1, 3, TopLeft()], "e"; left=8)
        plot_lr_partial_dependence!(gl[1, 3], model, chains)
        colsize!(gl, 3, Auto(0.5))
    end

    # Nudge layout
    colsize!(gl, 1, Auto(2))
    colsize!(f.layout, 1, Auto(2))
    colsize!(f.layout, 2, Auto(3))
    colgap!(gl, 3pt)
    rowgap!(f.layout, 6pt)
    resize_to_layout!(f)

    return f
end

function plot_all(dir; kwargs...)
    for model in readdir(dir; join=true)
        isfile(joinpath(model, "chains.jld2")) || continue
        @info "plotting $model"
        data = jldopen(joinpath(model, "chains.jld2"), "r")
        run_name = basename(model)
        figure_bayesian(data; kwargs...) |> MISTStyle.savefig("bayesian-$run_name")
        plot_prediction_intervals!(Figure(), data["model"], data["chains"]) |> MISTStyle.savefig("parity-$run_name")
        close(data)
    end
end
