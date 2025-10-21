#!/usr/bin/env -S julia +release --color=auto --startup-file=no --project=@script --threads=auto
# Script to fit neural scaling laws to dec3 sweep results
using BayesianScaling: BayesianScaling, HoffmanScaling
using DataFrames
using CSV: CSV
using Downloads: Downloads
using Enzyme: Enzyme
using JLD2: jldopen
using Setfield: @set!
using Distributions: Gamma

"""
    df = load_df_from_chains(file)

Load scaling data from a previously fit chains file
"""
function load_df_from_chains(file)
    model = jldopen(file, "r")["model"]
    df = DataFrame(BayesianScaling.observations(model))
    df.loss = BayesianScaling.response(model)
    return df
end

"""
    model, chains = fit_hoffmann(df)

Fit a Hoffmann scaling model to the data in `df`
"""
function fit_hoffmann(df)
    df = select(df, :model_size, :data_size, :loss)
    model = BayesianScaling.init_model(HoffmanScaling(), df)
    @set! model.priors.sigma = Gamma(2, 0.1)
    chains = BayesianScaling.sample_chains(model; nchains=4, draws=10000)
    return model, chains
end

"""
    df = select_best_runs(df)

Select the best run (lowest loss) for each model size and data size
"""
function select_best_runs(df)
    return combine(groupby(df, [:model_size, :data_size])) do gdf
        idx = argmin(gdf.loss)
        return gdf[idx, :]
    end
end

function (@main)(ARGS=[])
    file = ARGS[1]
    outdir = joinpath(pwd(), "out")

    df = load_df_from_chains(file)
    df_min = select_best_runs(df)

    # Fit baseline Hoffmann
    model, chains = fit_hoffmann(df)
    score=BayesianScaling.score_model(model, chains)
    @info "Baseline" score
    BayesianScaling.save_results(model, chains;
        outdir=joinpath(outdir, "hoffmann"),
        score,
    )

    # Repeat fit with best runs only
    model, chains = fit_hoffmann(df_min)
    score=BayesianScaling.score_model(model, chains)
    @info "Best runs" score
    BayesianScaling.save_results(model, chains;
        outdir=joinpath(outdir, "hoffmann--tuned-runs"),
        score,
    )

    return nothing
end
