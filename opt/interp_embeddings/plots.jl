#!/usr/bin/env -S julia --color=yes --startup-file=no --project=@script
using Makie
using DataFrames
using CSV: CSV
using JSON: JSON
using Random
using MISTStyle: MISTStyle, inch
using InterpEmb: figure_embedding, figure_token_embeddings

function arguments(args)
    options = Dict{String,String}()
    models = Pair{String,String}[]
    index = 1
    while index <= length(args)
        key = args[index]
        if key == "--model"
            index + 2 <= length(args) || error("--model requires a label and local directory")
            push!(models, args[index + 1] => args[index + 2])
            index += 3
        else
            index + 1 <= length(args) || error("$key requires a value")
            options[key] = args[index + 1]
            index += 2
        end
    end
    return options, models
end

function main(args)
    options, models = arguments(args)
    analysis, output = options["--analysis"], options["--output-dir"]
    mkpath(output)
    if analysis in ("token-main", "token-si", "token-paper")
        length(models) >= 3 || error("Provide pretrained, tmQM, and FreeSolv token artifacts first")
        settings = haskey(options, "--settings") ? JSON.parsefile(options["--settings"]) : Dict()
        allowed = ["seed", "p", "maxiter", "exploreiter", "exaggeration", "tol", "initialize"]
        isempty(setdiff(keys(settings), allowed)) || error("Unsupported t-SNE settings")
        kwargs = Dict(Symbol(key) => value for (key, value) in settings if key != "seed")
        haskey(kwargs, :initialize) && (kwargs[:initialize] = Symbol(kwargs[:initialize]))
        haskey(settings, "seed") && (kwargs[:rng] = MersenneTwister(settings["seed"]))
        kwargs = (; kwargs...)
        figure = with_theme(MISTStyle.theme()) do
            if analysis in ("token-main", "token-paper")
                figure_token_embeddings(models; emb_models=3, last_token=30,
                    show_updates=analysis != "token-paper",
                    fig_size=analysis == "token-paper" ? (240, 115) : (360, 115),
                    tsne_kwargs=kwargs, projection_dir=joinpath(output, "tsne"))
            else
                figure_token_embeddings(models; last_token=Inf, fig_size=(7inch, 4.5inch), min_update=1e-6,
                    tsne_kwargs=kwargs, projection_dir=joinpath(output, "tsne"))
            end
        end
        stem = analysis == "token-paper" ? "token_embeddings_paper" :
               analysis == "token-main" ? "token_embeddings_updates" : "token_embeddings_updates_si"
        MISTStyle.savefig(stem, figure; fig_dir=output)
    elseif analysis == "pah"
        benzene = DataFrame(CSV.File(options["--aromaticity"]))
        condensed = DataFrame(CSV.File(options["--condensation"]))
        with_theme(MISTStyle.theme()) do
            figure_embedding(benzene, condensed)
        end |> MISTStyle.savefig("interp_embeddings"; fig_dir=output)
        with_theme(MISTStyle.theme()) do
            figure_embedding(benzene, condensed; panel_labels=("c)", "", "d)"))
        end |> MISTStyle.savefig("interp_embeddings_paper"; fig_dir=output)
    else
        error("Unknown embedding analysis: $analysis")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
