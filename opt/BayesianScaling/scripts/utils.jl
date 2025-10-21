using BayesianScaling: BayesianScaling, sample_chains, save_results
using MCMCDiagnosticTools: ess_rhat
using JLD2: jldsave, jldopen
using UUIDs: uuid4
using Dates: Dates
using DataFrames: DataFrame
using Makie: with_theme

include("../../style.jl")
using .MISTStyle: MISTStyle, savefig


function save_figures(outdir::AbstractString; kwargs...)
    data = jldopen(joinpath(outdir, "chains.jld2"), "r")
    df = get(data, "df", DataFrame(data["model"].runs))
    save_figures(outdir, data["model"], df, data["chains"], data["raw_chains"]; kwargs...)
end

function save_figures(outdir, model, df, chains, raw_chains; n=500)
    # Subsample for faster plotting
    mkpath(outdir)
    chains = BayesianScaling.subsample(chains, n)

    # Generate Plots
    θ_scaling = selectdim(chains, 3, :scaling)
    with_theme(MISTStyle.theme()) do
        @sync begin
            Threads.@spawn savefig("scaling", BayesianScaling.plot_scaling(chains, df); fig_dir=outdir)
            Threads.@spawn savefig("lr_map", BayesianScaling.plot_lr_map(model, chains, df); fig_dir=outdir)
            Threads.@spawn savefig("compute_optimal", BayesianScaling.plot_compute_optimal(θ_scaling, df); fig_dir=outdir)
            Threads.@spawn savefig("penalties", BayesianScaling.plot_penalty(model, chains); fig_dir=outdir)
            Threads.@spawn savefig("summary", BayesianScaling.figure_ai4x(model, chains); fig_dir=outdir)

            # Bayesian Plots
            for sym in [:scaling, :lr]
                θ = selectdim(raw_chains, 3, sym)
                Threads.@spawn savefig("raw_chains_$sym", BayesianScaling.plot_chains(θ); fig_dir=outdir)
                Threads.@spawn savefig("raw_chains_covar_$sym", BayesianScaling.plot_chain_covariance(θ); fig_dir=outdir)
            end
        end
    end

    return outdir
end

function process_model(model, df::DataFrame)
    # Sample chains
    chains, chains_raw = sample_chains(model; adtype=:Enzyme)
    outdir = save_results(model, chains, chains_raw)
    @info "Saved results to $outdir"

    # Check convergence
    qoc = ess_rhat(chains_raw)
    @info "MCMC Convergence" qoc.ess qoc.rhat

    # Generate Plots
    save_figures(outdir, model, df, chains, chains_raw)

    return outdir
end
