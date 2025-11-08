function figure_ai4x(
    model,
    chains::AbstractArray{<:Real,3};
    N=logrange(1e5, 1e10; length=100),
    C=logrange(1e10, 1e26; length=100),
    p=0.95,
)
    df = DataFrame(model.runs)
    sort!(df, :loss; rev=true)
    loss_hat = expected_loss(model, chains)

    # Setup Plots
    f = Figure()
    gl_scaling = GridLayout(f[1, 2])
    gl_penalty = GridLayout(f[1, 1])
    rowgap!(gl_scaling, 5)
    rowgap!(gl_penalty, 5)

    # Compute Optimal Scaling
    ax_opt = Axis(gl_scaling[1, 1];
        yscale=log10,
        xscale=log10,
        limits=(extrema(C) ./ pf_day, (1e-3, 1)),
        xlabel="Compute Budget [PF-Days]",
        ylabel="Validation Loss",
    )
    μ, band = compute_optimal_loss(selectdim(chains, 3, :scaling), C; p)
    train_compute = @. 6 * float(df.data_size) * float(df.model_size) / pf_day
    scatter!(ax_opt, train_compute, df.loss; marker=:x, color=:red)
    predictionband!(ax_opt, C ./ pf_day, μ, band[1, :], band[2, :])

    # Scaling Plot
    ax_scale = Axis(gl_scaling[2, 1];
        xscale=log10,
        yscale=log10,
        limits=(extrema(C) ./ pf_day, extrema(N)),
        xlabel="Compute Budget [PF-Days]",
        ylabel="Model Size",
    )
    linkxaxes!(ax_opt, ax_scale)
    hidexdecorations!(ax_opt; grid=false)

    loss_mean, loss_band = hoffman_compute_scaling(selectdim(chains, 3, :scaling), N, C)
    levels = logrange(1e-3, 1; length=20)
    h = contourf!(ax_scale, C ./ pf_day, N, loss_mean'; levels, colorscale=log10)
    cb = Colorbar(gl_scaling[2, 2], h; label="Loss")

    N_opt, N_band = compute_optimal_model_size(selectdim(chains, 3, :scaling), C)
    predictionband!(ax_scale, C ./ pf_day, N_opt, selectdim(N_band, 1, 1), selectdim(N_band, 1, 2);
        color=:black,
        band_color=(:slategray, 0.4),
    )
    model_flops = @. 6 * float(df.model_size) * float(df.data_size) / pf_day
    scatter!(ax_scale, model_flops, df.model_size;
        color=df[:, :loss],
        colormap=cb.colormap,
        colorrange=cb.limits,
        colorscale=cb.scale,
        strokewidth=0.5,
        strokecolor=:black,
    )

    # Penalty Terms
    P = expected_penalties(model, chains)
    lr_ratios = logrange(1e-2, 1e2; length=50)
    ax = Axis(gl_penalty[1, 1];
        # xlabel="η/η₀",
        ylabel=L"P_{\eta}",
        xscale=log2,
        yscale=log2,
        limits=((1 / 128, 32), (0.25, 8)),
    )
    lr = df.lr
    lr_ideal = ideal_lr(model, chains)
    lr_ideal = vec(mean(lr_ideal; dims=(1, 2)))
    lr_resid = penalty_residual(df.loss, loss_hat, P.lr)

    lr_sense = expected_lr_sensitivity(model, chains, lr_ratios)
    scatter!(ax, lr ./ lr_ideal, lr_resid; marker=:x, color=:red)
    predictionband!(ax, lr_ratios, credible_interval(lr_sense; p)...)

    # FF Ratio
    ff = logrange(0.5, 20; base=2, length=100)
    ax = Axis(gl_penalty[2, 1];
        # xlabel="FF. Ratio",
        ylabel=L"P_{ff}",
        limits=(extrema(ff), (0.125, 16)),
        yscale=log2,
        xscale=log2,
    )
    df = DataFrame(model.runs)
    ff_resid = penalty_residual(df.loss, loss_hat, P.ff)
    ff_sense = harmonic_penalty_posterior(selectdim(chains, 3, :ff_ratio), ff) .|> exp
    predictionband!(ax, ff, credible_interval(ff_sense; p)...)
    scatter!(ax, df.ff_ratio, ff_resid; color=:red)

    # Aspect Ratio
    aspect = logrange(16, 200; base=2, length=100)
    ax = Axis(gl_penalty[3, 1];
        # xlabel="Aspect Ratio",
        ylabel=L"P_{a}",
        limits=(extrema(aspect), nothing),
        xscale=log2,
        yscale=log2,
    )
    df = DataFrame(model.runs)
    aspect_resid = penalty_residual(df.loss, loss_hat, P.aspect)
    sense = harmonic_penalty_posterior(selectdim(chains, 3, :aspect_ratio), aspect) .|> exp
    predictionband!(ax, aspect, credible_interval(sense; p)...)
    scatter!(ax, df.aspect_ratio, aspect_resid; color=:red)

    resize_to_layout!(f)
    return f
end
