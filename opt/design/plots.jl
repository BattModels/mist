#!/usr/bin/env -S uv run julia --project=@script --startup-file=no
using PythonCall
using Makie
using DesignRules
using MISTStyle
using DataFrames
using CSV: CSV
using JSON: JSON
using StatsBase: mean

include("olfaction_trends.jl")

function load_models(paths)
    loader = pyimport("electrolyte_fm.models.prod_finetune").load_finetuned
    return Dict(key => loader(path; local_files_only=true) for (key, path) in paths)
end

function design_plots(analysis, models, output, inputs; charges_path=nothing)
    if analysis == "olfaction"
        straight = predict_scent(models["odour"])
        branched = predict_scent(models["odour"], true)
        with_theme(MISTStyle.theme()) do
            plot_scent_by_type(straight)
        end |> MISTStyle.savefig("olfaction-trends"; fig_dir=output)
        with_theme(MISTStyle.theme()) do
            plot_branched_scent_comparison(branched)
        end |> MISTStyle.savefig("branched-olfaction"; fig_dir=output)
        return
    end
    qm = models["baseline"]
    if analysis == "hydrocarbons"
        for (variant, qm_model) in ("baseline" => qm, "randomized" => models["randomized"])
            predictors = (qm_model, models["mp"], models["bp"], models["fp"])
            hydrocarbons = DesignRules.predict_all(DesignRules.simple_hydrocarbons(30), predictors...; n=10)
            with_theme(MISTStyle.theme()) do
                DesignRules.hydrocarbon_trends(hydrocarbons)
            end |> MISTStyle.savefig("$variant-hydrocarbons"; fig_dir=output)
            if variant == "baseline"
                with_theme(MISTStyle.theme()) do
                    DesignRules.hydrocarbon_trends(hydrocarbons; paper_subset=true)
                end |> MISTStyle.savefig("paper-hydrocarbons"; fig_dir=output)
            end
            fats = DesignRules.predict_all(DesignRules.saturated_fats(24; n_max=9, d_max=6), predictors...; n=3)
            for omega in [3, 6, 9]
                with_theme(MISTStyle.theme()) do
                    DesignRules.figure_fatty_acids(fats; omega)
                end |> MISTStyle.savefig("$variant-omega-$omega-saturated-fats"; fig_dir=output)
            end
        end
    elseif analysis == "permutations"
        randomized = models["randomized"]
        molecules = DesignRules.simple_hydrocarbons(25)
        sample_smi(m) = :smi => ByRow(smi -> DesignRules.sample_encodings(smi, m)) => AsTable
        reference = transform(molecules, sample_smi(qm))
        augmented = transform(molecules, sample_smi(randomized))
        order = DesignRules.alkene_sweep(25, qm)
        random_order = DesignRules.alkene_sweep(25, randomized)
        comparisons = ("Baseline" => reference, "Random" => augmented)
        ordering = ("Baseline" => order, "Random" => random_order)
        with_theme(MISTStyle.theme()) do
            DesignRules.figure_permutations(comparisons...; name_df_order=ordering)
        end |> MISTStyle.savefig("permutations"; fig_dir=output)
        with_theme(MISTStyle.theme()) do
            DesignRules.figure_double_bond_loc(comparisons...; name_df_order=ordering)
        end |> MISTStyle.savefig("double_bond_loc"; fig_dir=output)
    elseif analysis == "electrolytes"
        molecules = DataFrame(CSV.File(joinpath(inputs, "electrolytes.csv"); stringtype=String))
        molecules.smi .= DesignRules.encode.(molecules.smi; encoding="smiles-kekule")
        predictors = (qm, "rand" => models["randomized"], "kt" => (models["kt"] => ["alpha", "beta"]),
                      "kt" => models["pka"], models["dn"], models["bp"], models["mp"], models["fp"])
        df = DesignRules.predict_all(molecules, predictors...; n=10)
        rename!(df, "BF3 affinity" => "dn", "DMSO_pKa_kt" => "pKa_kt")
        charges_path === nothing && error("--charges must name this model's PubChemQC predictions")
        charges = DesignRules.pubchem_from_jsonl(charges_path)
        charges.smi .= DesignRules.encode.(charges.smi; encoding="smiles-kekule")
        df = leftjoin(df, charges; on=:smi, validate=(false, true))
        any(ismissing, df.min_lowdin) && error("Partial-charge predictions do not cover every electrolyte")
        with_theme(MISTStyle.theme()) do
            DesignRules.electrolyte_trends(df)
        end |> MISTStyle.savefig("electrolytes"; fig_dir=output)
        transform(select(df, [:smi, :homo, :gap, :mp, :bp]),
            [:homo, :gap, :mp, :bp] .=> ByRow(mean) .=> [:homo, :gap, :mp, :bp]) |>
            CSV.write(joinpath(output, "electrolytes_predictions.csv"))
    else
        error("Unknown analysis: $analysis")
    end
end

function (@main)(args)
    option(name) = args[findfirst(==(name), args) + 1]
    model_paths = Dict{String,String}()
    for i in eachindex(args)
        args[i] == "--model" || continue
        model_paths[args[i + 1]] = args[i + 2]
    end
    output = option("--output-dir")
    mkpath(output)
    models = load_models(model_paths)
    analysis = option("--analysis")
    inputs = "--inputs" in args ? option("--inputs") : ""
    charges_path = "--charges" in args ? option("--charges") : nothing
    design_plots(analysis, models, output, inputs; charges_path)
    open(joinpath(output, "report.json"), "w") do io
        JSON.print(io, Dict("base" => option("--base"), "analysis" => analysis,
            "models" => model_paths, "figures" => filter(f -> endswith(f, ".pdf"), readdir(output))))
    end
end
