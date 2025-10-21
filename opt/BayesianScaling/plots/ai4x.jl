using Makie
using MISTStyle
using JLD2: jldopen
using StatsBase: mean
using Format: format
using BayesianScaling: BayesianScaling, compute_optimal_loss, predictionband!, hoffman_compute_scaling, compute_optimal_model_size, expected_penalties, expected_lr_sensitivity, credible_interval, pf_day

function load_prediction(dir)
    return jldopen(joinpath(dir, "chains.jld2")) do data
        data["model"].ℓ.log_density_function, data["chains"]
    end
end

function figure_ai4x(model, chains;
    N=logrange(1e5, 1e10; length=100),
    C=logrange(1e10, 2 * pf_day; length=100),
    B=logrange(2^2, 2^25; length=50),
    p=0.99,
)
    f = Figure(; size=(3inch, 2inch), figure_padding=(1, 1, 1, 4))

    chains = BayesianScaling.subsample(chains, 500)

    # Extract Empirical Data
    model_loss = map(run -> run.loss, model.runs)
    sdx = sortperm(model_loss; rev=true)
    permute!(model.runs, sdx)
    model_loss = model_loss[sdx]
    model_size = map(run -> run.model_size, model.runs)
    data_size = map(run -> run.data_size, model.runs)
    lr = map(run -> run.lr, model.runs)
    model_flops = @. 6 * float(model_size) * float(data_size) / pf_day

    # LR Penalty
    P = expected_penalties(model, chains)
    lr_ratios = logrange(1e-2, 1e2; length=50) |> collect
    ax = Axis(f[1, 1];
        xlabel=L"\eta/\eta_{*}",
        ylabel=L"P_{\eta}",
        xscale=log2,
        yscale=log2,
        limits=((1 / 128, 32), (0.5, 8)),
    )
    lr_ideal = BayesianScaling.ideal_lr(model, chains)
    lr_ideal = vec(mean(lr_ideal; dims=(1, 2)))
    loss_hat = BayesianScaling.expected_loss(model, chains)
    lr_resid = BayesianScaling.penalty_residual(model_loss, loss_hat, P.lr)

    lr_sense = expected_lr_sensitivity(model, chains, lr_ratios)
    scatter!(ax, lr ./ lr_ideal, lr_resid; marker=:circle, color=:red, markersize=2pt)
    predictionband!(ax, lr_ratios, credible_interval(lr_sense; p)...)


    # Compute Optimal Frontier
    ax_scale = Axis(f[1, 2];
        xscale=log10,
        yscale=log10,
        limits=(extrema(C) ./ pf_day, extrema(N)),
        xlabel="Compute Budget [PF-Days]",
        ylabel="Model Size",
    )

    loss_mean, loss_band = hoffman_compute_scaling(selectdim(chains, 3, :scaling), collect(N), collect(C))
    levels = logrange(5e-4, 1; length=20)
    h = contourf!(ax_scale, C ./ pf_day, N, selectdim(loss_band, 1, 2)'; levels, colorscale=log10)
    cb = Colorbar(f[1, 3], h; label="Loss")

    N_opt, N_band = compute_optimal_model_size(selectdim(chains, 3, :scaling), collect(C))
    predictionband!(ax_scale, C ./ pf_day, N_opt, selectdim(N_band, 1, 1), selectdim(N_band, 1, 2);
        color=RGBf(1.0, 0.796, 0.02),
        band_color=(RGBf(0.596, 0.612, 0.592), 0.4),
    )
    h = scatter!(ax_scale, model_flops, model_size;
        marker=:circle,
        color=model_loss,
        colormap=cb.colormap,
        colorrange=cb.limits,
        colorscale=cb.scale,
        strokewidth=0.5,
        strokecolor=:black,
        markersize=3pt,
    )

    N_4yzwys2z = BayesianScaling.non_embedding_size(1024, 4 * 1024, 18)
    C_4yzwys2z = 6 * N_4yzwys2z * 374685 * 4 * 8 * 16 * 32 / pf_day
    scatter!(ax_scale, C_4yzwys2z, N_4yzwys2z;
        color=0.01019080262631178,
        marker=:star5,
        colormap=cb.colormap,
        colorrange=cb.limits,
        colorscale=cb.scale,
        strokewidth=h.strokewidth,
        strokecolor=h.strokecolor,
        markersize=@lift(2 * $(h.markersize)),
    )


    lr_quantiles = BayesianScaling.ideal_lr_map(model, chains, collect(N), collect(B); p=0.95)
    lr_mean = selectdim(lr_quantiles, 3, 1)
    lr_uq = 0.5 * (selectdim(lr_quantiles, 3, 1) - selectdim(lr_quantiles, 3, 2)) ./ lr_mean

    ax = Axis(
        f[2, 1:2];
        limits=(extrema(N), extrema(B)),
        xlabel="Model Size",
        ylabel="Batch Size",
        xscale=log10,
        yscale=log10,
    )
    h = contourf!(ax, N, B, lr_mean;
        colorscale=log10,
        levels=logrange(extrema(lr_mean)...; length=20)
    )
    contour!(ax, N, B, lr_uq;
        color=:white,
        labels=true,
        levels=[0.125, 0.25, 0.5],
        labelsize=5pt,
        linewidth=0.5,
        labelformatter=x -> format("{:.0%}", x)
    )
    Colorbar(f[2, 3], h; label=L"\eta_*")




    label_kwargs = (;
        fontsize=6pt,
        font=:bold,
        halign=:right,
        tellheight=false,
        padding=(0, 23, 0, 0)
    )
    Label(f[1, 1, TopLeft()], "a)"; label_kwargs...)
    Label(f[1, 2, TopLeft()], "b)"; label_kwargs...)
    Label(f[2, 1, TopLeft()], "c)"; label_kwargs...)


    resize_to_layout!(f)
    return f

end
