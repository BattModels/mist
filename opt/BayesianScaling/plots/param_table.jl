using PrettyTables
using DataFrames
using BayesianScaling: BayesianScaling, find, selectparam

scaling(x) = haskey(x[1,1,:], :scaling) ? selectdim(x, 3, :scaling) : x

function model_summary(files)
    rows = []
    for file in files
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
        ("Penalized", :d_model, true, false) => latex_cell"Penalized, Scaling with $d_{model}$",
        ("Penalized", :model_size, false, true) => latex_cell"Penalized, Additive",
        ("Penalized", :model_size, true, true) => latex_cell"Penalized, Harmonic Shape Penalty",
        ("Chinchilla", nothing, nothing, nothing) => "Chinchilla",
    )
    return get(named, (model, lr_model_size, geometric_penalty, harmonic_shape_penalty), nothing)
end

function sn(x::Real; sigdigits::Int=3)
    if x == 0
        return "0.0"
    end

    str = format(x; precision=sigdigits-1, conversion="e")

    # Split mantissa and exponent
    m, e = split(str, 'e')
    e = replace(e, r"^\+?" => "")  # remove optional '+'
    e = replace(e, r"^0+" => "")   # remove leading zeros

    m = replace(m, r"\.?0+$" => "")  # strip trailing .00

    return "\\sn{$m}{$e}"
end

function write_scaling_param_table(df::DataFrame; sigdigits=3)
    df = transform(df, [:model, :lr_model_size, :geometric_penalty, :harmonic_shape_penalty] => ByRow(model_name) => :model_name)
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

function (@main)(ARGS)
    df = model_summary(BayesianScaling.find("out", r"chains.jld2"))
    subset!(df, :n_smooth => ByRow(==(1)), :file => ByRow(endswith("gamma")))
    table = write_scaling_param_table(df)
    open(joinpath("fig", "param_table.tex"), "w") do io
        write(io, table)
    end
end
