#!/usr/bin/env -S uv run julia --project=@script --startup-file=no
using CSV
using DataFrames
using Makie
using MISTStyle
using Statistics
using JSON

function arguments(args)
    length(args) % 2 == 0 || error("Expected --option value pairs")
    options = Dict(args[i] => args[i + 1] for i in 1:2:length(args))
    haskey(options, "--analysis") || error("--analysis is required")
    haskey(options, "--output-dir") || error("--output-dir is required")
    return options
end

function finite_table(path, columns)
    isfile(path) || error("Supply the original olfaction input at $path")
    data = CSV.read(path, DataFrame)
    isempty(data) && error("Empty olfaction input: $path")
    isempty(setdiff(columns, names(data))) || error("$path requires columns $(join(columns, ", "))")
    all(isfinite, Matrix{Float64}(select(data, columns))) || error("Nonfinite values in $path")
    return data
end

function geometry_plots(input, output)
    # Numeric fits are external inputs. Do not rerun a stochastic fit and label
    # the result as the paper's geometry, or draw only a diagnostic scatter.
    for metric in ("euclidean", "hyperbolic")
        columns = ["radius"; ["$(kind)_betti_$(n)" for kind in ("empirical", "model") for n in 1:3]]
        curves = finite_table(joinpath(input, "$(metric)_betti.csv"), columns)
        issorted(curves.radius) || error("Betti filtration radii must be sorted")
        fig = Figure(size=(160mm, 50mm))
        for n in 1:3
            ax = Axis(fig[1, n]; xlabel="Filtration radius", ylabel="β$n")
            lines!(ax, curves.radius, curves[!, "empirical_betti_$n"]; label="Empirical")
            lines!(ax, curves.radius, curves[!, "model_betti_$n"]; label=titlecase(metric))
            n == 1 && axislegend(ax; position=:rt, labelsize=6pt)
        end
        MISTStyle.savefig("betti_curve_$metric", fig; fig_dir=output)
    end
    coords = finite_table(joinpath(input, "hydra_coordinates.csv"), ["dimension_0", "dimension_1", "dimension_2"])
    ids = CSV.read(joinpath(input, "molecule_ids.csv"), DataFrame)
    coords.molecule_id == ids.molecule_id || error("HYDRA coordinates must preserve the original molecule order")
    length(unique(coords.molecule_id)) == nrow(coords) || error("HYDRA molecule IDs must be unique")
    xyz = Matrix{Float64}(select(coords, :dimension_0, :dimension_1, :dimension_2))
    span = maximum(abs, xyz)
    span > 0 || error("Degenerate HYDRA coordinates")
    colors = [RGBf((row ./ span .+ 1) ./ 2...) for row in eachrow(xyz)]
    fig = Figure(size=(65mm, 65mm))
    ax = Axis(fig[1, 1]; aspect=1)
    hidedecorations!(ax)
    hidespines!(ax)
    # Paper view: x/y projection, rectangles behind the shell, circles in front,
    # and RGB proportional to the supplied x/y/z coordinates.
    for (mask, marker) in ((xyz[:, 3] .< 0, :rect), (xyz[:, 3] .>= 0, :circle))
        scatter!(ax, xyz[mask, 1], xyz[mask, 2]; color=colors[mask], marker, markersize=3pt)
    end
    MISTStyle.savefig("olfaction-hyperbolic", fig; fig_dir=output)
end

function mixture_plot(input, output)
    data = finite_table(input, ["target", "prediction"])
    nrow(data) >= 3 || error("Mixture correlation requires at least three held-out pairs")
    r = cor(data.target, data.prediction)
    isfinite(r) || error("Mixture correlation is undefined")
    fig = Figure(size=(65mm, 60mm))
    ax = Axis(fig[1, 1]; xlabel="Measured perceptual distance", ylabel="Predicted perceptual distance",
              title="Pearson r = $(round(r; digits=3))", limits=((0, 1), (0, 1)), aspect=1)
    lines!(ax, [0, 1], [0, 1]; color=:gray, linestyle=:dash)
    scatter!(ax, data.target, data.prediction; markersize=3pt, color=(MISTStyle.UM_COLORS.blue, 0.6))
    MISTStyle.savefig("olfaction-mixtures", fig; fig_dir=output)
    CSV.write(joinpath(output, "mixture-performance.csv"), DataFrame(n=[nrow(data)], pearson=[r]))
end

function main(args)
    options = arguments(args)
    analysis, output = options["--analysis"], options["--output-dir"]
    mkpath(output)
    if analysis == "overview"
        ENV["OLFACTION_DATA_DIR"] = options["--input-dir"]
        include(options["--library"])
        Base.invokelatest() do
            with_theme(MISTStyle.theme()) do
                olfactory_panel()
            end |> MISTStyle.savefig("olfactory"; fig_dir=output)
        end
    elseif analysis == "discordance"
        include(options["--library"])
        Base.invokelatest(plot_discordance, options["--mapping"], options["--triplets"];
            molecule_path=options["--gnn"], model_path=options["--model"], output_dir=output)
    elseif analysis == "dendrogram"
        include(options["--library"])
        Base.invokelatest() do
            with_theme(MISTStyle.theme()) do
                _, X, labels = load_logits(options["--predictions"])
                # The command-line example retains its historical defaults. The
                # paper specifies sqrt(2*(1-rho)) with average linkage.
                distance = corr_distance_scents(X; metric=:paper)
                fig, _, _, _ = plot_radial_dendrogram(distance, labels; linkage=:average, k=12, figsize=(510, 500))
                MISTStyle.savefig("olfaction-clusters", fig; fig_dir=output)
            end
        end
    elseif analysis == "geometry"
        with_theme(MISTStyle.theme()) do
            geometry_plots(options["--input-dir"], output)
        end
    elseif analysis == "mixture"
        with_theme(MISTStyle.theme()) do
            mixture_plot(options["--predictions"], output)
        end
    else
        error("Unknown olfaction analysis: $analysis")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
