"""
    predictionband(center, lower, upper)

Plots a prediction band with median line overlayed
"""
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
    lr_eff(N; lr_0, lr_n, kwargs...) = @. exp(lr_0 - lr_n * log(N))
    s = sample_posterior(lr_eff, post_model, N')
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

function plot_scaling(chains, df;
    N=logrange(1e4, 1e12; length=200),
    C=logrange(1e8, 1e26; length=200)
)
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
    llm_scaling(n, d; A, α, B, β, E, kwargs...) = @. (A / n^α) + (B / d^β) + E
    for (idx, n) in enumerate(N)
        d = @. C / (6n)
        loss[:, idx] = median(sample_posterior(llm_scaling, chains, n, d); dims=1)
    end
    loss_low = min(minimum(df.loss), minimum(loss))
    loss_high = maximum(df.loss)
    levels = collect(logrange(loss_low, loss_high; length=20))
    h = contourf!(ax, C, N, loss;
        levels,
        colorscale=log10,
    )
    contour!(ax, C, N, loss; levels=[0.06], color=:red)
    # Colorbar(f[1, 2], h; label="Validation Loss")

    # Add Empirical Loss
    df.flops = @. 6 * df.N * df.D
    h = scatter!(ax, df.flops, df.N;
        color=df.loss,
        colormap=h.colormap,
        colorrange=@lift(extrema($(h.levels))),
        colorscale=h.colorscale,
        strokewidth=0.5,
        strokecolor=:black,
        label="Emperical Data"
    )
    Colorbar(f[1, 2], h; label="Finetuning Loss, Shear Modulus")

    # Add Compute Optimal Frontier
    s = sample_posterior(compute_optimal_model, chains, C)
    q = map(c -> quantile(c, (0.5, 0.025, 0.975)), eachcol(s))
    predictionband!(ax, C, [x[1] for x in q], [x[2] for x in q], [x[3] for x in q];
        color=:black,
        band_color=(:slategray, 0.4),
        label="Compute Optimal Frontier",
    )
    # hlines!(ax, 29491200; label="MolFormer", color=:red)
    # vlines!(ax, 60 * 60 * 44e15; label="Polaris-hour", color=:purple)
    # vlines!(ax, 60 * 60 * 2.5e12; label="Artemis-hour", color=:blue)

    # f[2, 1] = Legend(f, ax; tellheight=true, tellwidth=false, orientation=:horizontal, nbanks=2)


    return f
end

function plot_parity(model, chains; p=0.025)
    loss_samples = hcat(generated_quantities(model, chains)...)
    loss_median = vec(median(loss_samples; dims=2))
    loss_low = vec(map(Base.Fix2(quantile, p), eachrow(loss_samples)))
    loss_high = vec(map(Base.Fix2(quantile, 1 - p), eachrow(loss_samples)))
    ll = min(minimum(model.args.loss), minimum(loss_low)) * 0.9
    uu = max(maximum(model.args.loss), maximum(loss_high)) * 1.1

    f = Figure()
    ax = Axis(f[1, 1];
        yscale=log10,
        xscale=log10,
        ylabel="Predicted Loss",
        xlabel="Measured Loss",
        limits=((ll, uu), (ll, uu)),
    )
    rangebars!(ax, model.args.loss, loss_low, loss_high; color=:blue, linewidth=1)
    # scatter!(ax, model.args.loss, loss_median; color=:black, marker=:x, markersize=5)

    # Plot Parity line
    n = 100
    lines!(ax, range(ll, uu; length=n), range(ll, uu; length=n), color=:red)
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


function plot_acquisition(chains, aq;
    N=logrange(1e5, 1e12; length=50),
    D=logrange(1e6, 10e15; length=50)
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
    Colorbar(f[1, 2], h; label="Acquisition Function")
    return f
end
