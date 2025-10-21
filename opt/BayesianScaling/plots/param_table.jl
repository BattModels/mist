#!/usr/bin/env -S julia +release --color=auto --startup-file=no --project=@script
using PrettyTables
using DataFrames
using BayesianScaling: BayesianScaling, find, selectparam
using JLD2: jldopen
using MISTStyle
using Makie

scaling(x) = haskey(x[1,1,:], :scaling) ? selectdim(x, 3, :scaling) : x

include("utils.jl")

function model_summary(files)
    rows = []
    for file in files
        @info file
        data = jldopen(file, "r")
        score = data["score"]
        summary_params = BayesianScaling.scaling_summary(scaling(data["chains"]))
        scale_params = BayesianScaling.credible_interval(scaling(data["chains"]))

        # Compute expected loss
        model = data["model"]
        μ = first(BayesianScaling.predict_ci(model, data["chains"]))
        loss = response(model)

        # Record smoothing
        n_smooth = parse(Float64, match(r"smoothed-(.*?)--", file).captures[1])

        # Record model info
        if model.formula isa BayesianScaling.HoffmanScaling
            model_info = (;
                model = "Chinchilla",
                lr_model_size = nothing,
                geometric_penalty = nothing,
                harmonic_shape_penalty = nothing,
                eta_0 = nothing,
                eta_batch_size = nothing,
                eta_model_size = nothing,
            )
        else
            chains = data["chains"]
            eta_0 = selectparam(chains, :lr, :ideal, :a) |> BayesianScaling.credible_interval
            eta_batch_size = selectparam(chains, :lr, :ideal, :b) |> BayesianScaling.credible_interval
            eta_model_size = selectparam(chains, :lr, :ideal, :c) |> BayesianScaling.credible_interval
            model_info = (;
                model = "Penalized",
                lr_model_size = model.formula.lr_model_size,
                geometric_penalty = model.formula.geometric_penalty,
                harmonic_shape_penalty = model.formula.harmonic_shape_penalty,
                eta_0,
                eta_batch_size,
                eta_model_size,
            )
        end

        push!(rows, (;
            file=basename(dirname(file)),
            model_info...,
            n_smooth,
            score...,
            summary_params...,
            scale_params...,
            pearson=cor(μ, loss),
            spearman=corspearman(μ, loss),
        ))
        close(data)
    end
    return DataFrame(rows)
end

function model_name(model, lr_model_size, geometric_penalty, harmonic_shape_penalty)

    named = Dict(
        ("Penalized", :model_size, true, false) => "Penalized, Baseline",
        ("Penalized", :d_model, true, false) => "Penalized, LR Scales with Hidden Size",
        ("Penalized", :model_size, false, true) => "Penalized, Additive",
        ("Penalized", :model_size, true, true) => "Penalized, Harmonic Shape Penalty",
        ("Chinchilla", nothing, nothing, nothing) => "No Penalties",
    )
    return get(named, (model, lr_model_size, geometric_penalty, harmonic_shape_penalty), nothing)
end

function write_scaling_param_table(df::DataFrame; sigdigits=3)
    df = transform(df, [:model, :lr_model_size, :geometric_penalty, :harmonic_shape_penalty] => ByRow(LatexCell∘model_name) => :model_name)
    subset!(df, :model_name => ByRow(!isnothing))
    sort!(df, :waic)
    round_sigfigs = ByRow(x -> round(x ; sigdigits))
    cols = [
        :model_name => "Model",
        :waic => latex_cell"\acs{WAIC}",
        :mape =>latex_cell"\acs{MAPE}",
        # :aic =>latex_cell"\acs{AIC}",
        :pearson =>latex_cell"Pearson's \(\rho\)",
        :spearman =>latex_cell"Spearman's \(\rho\)",
        :A =>latex_cell"$A$",
        :B =>latex_cell"$B$",
        :α =>latex_cell"$\alpha$",
        :β =>latex_cell"$\beta$",
        :E =>latex_cell"$E \cdot 1000$",
        # :ζ =>latex_cell"$\frac{\alpha\beta}{\alpha\beta}$",
        :r => latex_cell"$\frac{\alpha}{\beta}$",
    ]
    tf = LatexTableFormat(;
        header_envs=[],
        subheader_envs=[],
    )
    function fmt_ci(v, i, j)
        col = first(cols[j])
        col in [:A, :B, :α, :β, :E, :ζ, :r] || return v
        if col in [:E]
            v = v .* 1000
        end

        μ, l, u = v
        μ = round(μ; sigdigits)
        l = round(l; sigdigits)
        u = round(u; sigdigits)
        if col in [:A, :B]
            μ = sn(μ; sigdigits)
            l = sn(l; sigdigits)
            u = sn(u; sigdigits)
        end
        return LatexCell("\\ci{$μ}{$l}{$u}")
    end
    function fmt_metric(v, i, j)
        metrics = [:waic, :mape, :aic, :pearson, :spearman]
        first(cols[j]) in metrics || return v
        metric = first(cols[j])
        v = round(v; sigdigits)
        if metric == :waic
            return format("{:d}", v)
        elseif metric == :mape
            return format("{:.1%}", v)
        else
            return format("{}", v)
        end
    end
    return pretty_table(df[:, first.(cols)];
        tf,
        alignment=[col in [:model_name] ? :l : :c for col in first.(cols)],
        formatters=(fmt_ci, fmt_metric),
        backend=Val(:latex),
        hlines=[:header],
        header=last.(cols),
        wrap_table=false,
    )
end

function figure_smoothed_scaling(df)
    sort!(df, :n_smooth)
    transform!(df, [:model, :lr_model_size, :geometric_penalty, :harmonic_shape_penalty] => ByRow(model_name) => :model_name)
    df = subset(df, :model_name => ByRow(!isnothing), :file => ByRow(endswith("gamma")))

    f = Figure(; size=(5inch, 3inch))
    ax_waic = Axis(f[1, 1];
        limits=(extrema(df.n_smooth), nothing),
        xlabel="Validation Observation Averaged Over",
        ylabel="WAIC",
        xscale=log10,
    )
    ax_rho = Axis(f[2, 1];
        limits=(extrema(df.n_smooth), (0, 1)),
        xlabel="Validation Observation Averaged Over",
        ylabel=L"Spearman's $\rho$",
        xscale=ax_waic.xscale,
    )
    ax_mape = Axis(f[3, 1];
        limits=(extrema(df.n_smooth), (0, nothing)),
        xlabel="Validation Observation Averaged Over",
        ylabel="MAPE",
        ytickformat="{:.0%}",
        xscale=ax_waic.xscale,
    )
    hidexdecorations!(ax_waic; grid=false)
    hidexdecorations!(ax_rho; grid=false)
    linkxaxes!(ax_waic, ax_rho, ax_mape)
    rowgap!(f.layout, 2, 6pt)

    linewidth=1pt
    for gdf in groupby(df, :model_name)
        @info first(gdf.model_name)
        lines!(ax_waic, gdf.n_smooth, gdf.waic; label=first(gdf.model_name), linewidth)
        lines!(ax_mape, gdf.n_smooth, gdf.mape; label=first(gdf.model_name), linewidth)
        lines!(ax_rho, gdf.n_smooth, gdf.spearman; label=first(gdf.model_name), linewidth)
    end

    Legend(f[4, 1], ax_rho;
        orientation=:horizontal,
        tellwidth=false,
        tellheight=true,
    )

    return f
end


function (@main)(ARGS)
    df = model_summary(BayesianScaling.find("out", r"dec-3-.*chains.jld2"))
    subset!(df, :n_smooth => ByRow(==(1)), :file => ByRow(endswith("gamma")))
    table = write_scaling_param_table(df)
    open(joinpath("fig", "param_table.tex"), "w") do io
        write(io, table)
    end
end
