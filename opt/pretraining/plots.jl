#!/usr/bin/env -S julia --startup-file=no --project=@script
# Render the paper's pretraining SI from original observations, without training.
using CSV
using DataFrames
using Makie
using MISTStyle

const ABLATION_TASKS = ["bbbp", "clintox", "esol", "freesolv", "qm9", "bace", "tox21", "toxcast"]
const ABLATION_VARIANTS = [
    ("Smirk-GPE", "Absolute"), ("Smirk-GPE", "Rotary"),
    ("Smirk", "Absolute"), ("Smirk-SELFIES", "Rotary"),
]

function original_table(path, columns; numeric=[])
    isfile(path) || error("Missing original pretraining observations: $path. See opt/pretraining/README.md for the required schema.")
    df = CSV.read(path, DataFrame)
    isempty(df) && error("No original observations in $path")
    absent = setdiff(columns, names(df))
    isempty(absent) || error("$path requires columns: $(join(absent, ", "))")
    any(ismissing, Matrix(select(df, columns))) && error("Missing values in $path")
    for column in numeric
        all(value -> value isa Real && isfinite(value), df[!, column]) || error("$path requires finite numeric $column")
    end
    return df
end

function unique_rows(df, columns, label)
    nrow(unique(select(df, columns))) == nrow(df) || error("Ambiguous duplicate $label rows for $(join(columns, ", "))")
end

function transfer_observations(path; samples_per_batch=8)
    samples_per_batch > 0 || error("samples_per_batch must be positive")
    columns = ["Name", "cli.model.init_args.freeze_encoder", "cli.trainer.limit_train_batches",
               "cli.model.init_args.from_pretrained", "cli.model.init_args.encoder_ckpt", "val/loss_epoch.min"]
    source = original_table(path, columns)
    sizes = Dict("ti624ev1" => "28M", "dh61satt" => "1.8B")
    rows = NamedTuple[]
    for row in eachrow(source)
        matches = [size for (id, size) in sizes if id in split(row["cli.model.init_args.encoder_ckpt"], '/')]
        isempty(matches) && continue  # The released 228M runs are not in this paper figure.
        length(matches) == 1 || error("Ambiguous transfer encoder for $(row.Name)")
        frozen = row["cli.model.init_args.freeze_encoder"]
        pretrained = row["cli.model.init_args.from_pretrained"]
        frozen isa Bool && pretrained isa Bool || error("Transfer policy columns must contain true/false")
        batches = row["cli.trainer.limit_train_batches"]
        batches isa Integer && batches > 0 || error("Transfer data limits must be positive batch counts")
        loss = row["val/loss_epoch.min"]
        loss isa Real && isfinite(loss) && loss >= 0 || error("Transfer losses must be finite and nonnegative")
        push!(rows, (; run_name=row.Name, model_size=only(matches), frozen, pretrained,
                     samples=batches * samples_per_batch, loss))
    end
    isempty(rows) && error("No paper 28M/1.8B transfer runs in $path")
    df = DataFrame(rows)
    unique_rows(df, ["model_size", "frozen", "pretrained", "samples"], "transfer")
    for size in ["28M", "1.8B"], frozen in [false, true], pretrained in [false, true]
        count((df.model_size .== size) .& (df.frozen .== frozen) .& (df.pretrained .== pretrained)) >= 2 ||
            error("Missing transfer curve: $size, frozen=$frozen, pretrained=$pretrained")
    end
    return sort(df, [:frozen, :model_size, :pretrained, :samples])
end

function transfer_figure(df)
    # Reuses opt/pretrain_adv.jl from de0268fb3d57178ef3bbe293e48f063110c58bdf:
    # one scatterline per model/initialization, circles pretrained, rectangles random.
    # The paper arranges the same comparison into frozen/unfrozen panels.
    figure = Figure(; size=(3.25inch, 4inch), figure_padding=(5pt, 8pt, 2pt, 2pt))
    axes = Axis[]
    colors = Dict("28M" => MISTStyle.CAT_COLORS[2], "1.8B" => MISTStyle.CAT_COLORS[3])
    for (index, frozen) in enumerate([false, true])
        axis = Axis(figure[index, 1]; xlabel="Number of Samples", ylabel="Min. Val. Loss")
        push!(axes, axis)
        for group in groupby(subset(df, :frozen => ByRow(==(frozen))), [:model_size, :pretrained])
            scatterlines!(axis, group.samples, group.loss;
                color=colors[first(group.model_size)], marker=first(group.pretrained) ? :circle : :rect,
                linewidth=2pt, markersize=4pt)
        end
        sublabel!(figure[index, 1, TopLeft()], index == 1 ? "a" : "b"; left=8)
    end
    hidexdecorations!(axes[1]; grid=false)
    linkxaxes!(axes...)
    elements = Any[LineElement(color=colors[size], linewidth=2pt) for size in ["28M", "1.8B"]]
    append!(elements, [MarkerElement(color=:black, marker=marker) for marker in [:circle, :rect]])
    Legend(figure[0, 1], elements, ["MIST-28M", "MIST-1.8B", "Pretrained", "Not Pretrained"];
           orientation=:horizontal, nbanks=2, tellheight=true, tellwidth=false)
    return figure
end

function gas_observations(path)
    columns = ["run_id", "gpus", "gas", "microbatch_size", "samples_per_second"]
    df = original_table(path, columns; numeric=columns[2:end])
    all(df.microbatch_size .== 512) || error("Paper GAS measurements use 512 samples per GPU microbatch")
    all(df.samples_per_second .> 0) || error("Training throughput must be positive")
    unique_rows(df, ["gpus", "gas"], "selected throughput")
    expected = Set((gpus, gas) for gpus in [64, 256, 512] for gas in [1, 64])
    Set(zip(df.gpus, df.gas)) == expected || error("Supply the six original 64/256/512-GPU observations for GAS=1 and GAS=64; normalized logs through 64 GPUs are insufficient")
    return sort(df, [:gas, :gpus])
end

function gas_figure(df)
    figure = Figure(; size=(3.25inch, 2.1inch), figure_padding=(3pt, 12pt, 2pt, 2pt))
    axis = Axis(figure[1, 1]; xlabel="# GPUs", ylabel="Train Throughput [samples/s]",
                xscale=log2, xticks=([64, 128, 256, 512], string.([64, 128, 256, 512])))
    for gas in [64, 1]
        rows = subset(df, :gas => ByRow(==(gas)))
        scatterlines!(axis, rows.gpus, rows.samples_per_second;
                      label=gas == 1 ? "Without GAS" : "With GAS = 64", linewidth=2pt)
    end
    axislegend(axis; position=:lt)
    return figure
end

function instability_observations(loss_path, eigenvalue_path)
    losses = original_table(loss_path, ["run_id", "condition", "step", "train_loss", "spike_step"];
                            numeric=["step", "train_loss", "spike_step"])
    eigenvalues = original_table(eigenvalue_path,
        ["run_id", "condition", "step", "matrix_group", "layer", "matrix", "lambda_max"];
        numeric=["step", "layer", "lambda_max"])
    unique_rows(losses, ["condition", "step"], "training loss")
    unique_rows(eigenvalues, ["condition", "step", "matrix_group", "layer", "matrix"], "eigenvalue")
    Set(losses.condition) == Set(eigenvalues.condition) == Set(["Spiking", "Corrected"]) ||
        error("Both original Spiking and Corrected runs are required")
    Set(eigenvalues.matrix_group) == Set(["Feedforward", "KQV"]) || error("Original Feedforward and KQV eigenvalue distributions are required")
    length(unique(losses.spike_step)) == 1 || error("Provide the original common spike marker in spike_step")
    all(losses.train_loss .>= 0) && all(eigenvalues.lambda_max .>= 0) && all(losses.step .>= 0) && all(eigenvalues.step .>= 0) || error("Losses, eigenvalues, and steps must be nonnegative")
    for condition in ["Spiking", "Corrected"]
        ids = unique(losses.run_id[losses.condition .== condition])
        length(ids) == 1 || error("Select one original $condition training run")
        Set(eigenvalues.run_id[eigenvalues.condition .== condition]) == Set(ids) || error("Eigenvalues and loss must refer to the same $condition run")
        for group in ["Feedforward", "KQV"]
            subset_rows = eigenvalues[(eigenvalues.condition .== condition) .& (eigenvalues.matrix_group .== group), :]
            nrow(subset_rows) > 0 || error("Missing $condition/$group eigenvalue observations")
            for distribution in groupby(subset_rows, :step)
                nrow(distribution) >= 2 || error("Eigenvalue distributions require per-layer observations, not just a mean")
            end
        end
    end
    return sort(losses, [:condition, :step]), sort(eigenvalues, [:matrix_group, :step, :condition, :layer, :matrix])
end

function instability_figure(losses, eigenvalues)
    figure = Figure(; size=(4inch, 5inch), figure_padding=(5pt, 8pt, 2pt, 2pt))
    colors = [MISTStyle.CAT_COLORS[1], MISTStyle.CAT_COLORS[2]]
    axes = Axis[]
    marker = only(unique(losses.spike_step))
    for (index, group) in enumerate(["Feedforward", "KQV"])
        axis = Axis(figure[index, 1]; ylabel="λmax — $group")
        push!(axes, axis)
        rows = subset(eigenvalues, :matrix_group => ByRow(==(group)))
        groups = [condition == "Spiking" ? 1 : 2 for condition in rows.condition]
        boxplot!(axis, rows.step, rows.lambda_max; dodge=groups, color=groups, colormap=colors, colorrange=(1, 2))
        vlines!(axis, [marker]; color=:black, linestyle=:dash)
        hidexdecorations!(axis; grid=false)
    end
    axis = Axis(figure[3, 1]; xlabel="Training Steps", ylabel="Training Loss")
    push!(axes, axis)
    for (index, condition) in enumerate(["Spiking", "Corrected"])
        rows = subset(losses, :condition => ByRow(==(condition)))
        lines!(axis, rows.step, rows.train_loss; color=colors[index], linewidth=2pt, linestyle=:solid)
    end
    vlines!(axis, [marker]; color=:black, linestyle=:dash)
    linkxaxes!(axes...)
    Legend(figure[0, 1], [PolyElement(color=color) for color in colors], ["Spiking", "Corrected"];
           orientation=:horizontal, tellheight=true, tellwidth=false)
    sublabel!(figure[1, 1, TopLeft()], "a"; left=8)
    sublabel!(figure[3, 1, TopLeft()], "b"; left=8)
    return figure
end

function ablation_observations(path)
    df = original_table(path, ["run_id", "dataset", "tokenizer", "position_embedding", "metric", "value"]; numeric=["value"])
    unique_rows(df, ["dataset", "tokenizer", "position_embedding"], "selected ablation")
    expected = Set((dataset, tokenizer, position) for dataset in ABLATION_TASKS for (tokenizer, position) in ABLATION_VARIANTS)
    Set(zip(df.dataset, df.tokenizer, df.position_embedding)) == expected || error("Supply exactly the original eight tasks × four tokenizer/position variants; no unrecorded run selection is performed")
    for row in eachrow(df)
        metric = row.dataset in ["esol", "freesolv", "qm9"] ? "r2" : "auroc"
        row.metric == metric || error("$(row.dataset) requires validation $metric, not $(row.metric)")
        metric == "auroc" && !(0 <= row.value <= 1) && error("AUROC must be in [0,1]")
        metric == "r2" && row.value > 1 && error("R² cannot exceed 1")
    end
    return df
end

function ablation_figure(df)
    figure = Figure(; size=(6.5inch, 2.3inch), figure_padding=(3pt, 5pt, 2pt, 2pt))
    axis = Axis(figure[1, 1]; ylabel="Validation R² / AUROC",
                xticks=(1:8, ["BBBP", "ClinTox", "ESOL", "FreeSolv", "QM9", "BACE", "Tox21", "ToxCast"]),
                xticklabelrotation=π / 2, xgridvisible=false)
    task_ids = [findfirst(==(dataset), ABLATION_TASKS) for dataset in df.dataset]
    variant_ids = [findfirst(==((tokenizer, position)), ABLATION_VARIANTS) for (tokenizer, position) in zip(df.tokenizer, df.position_embedding)]
    colors = [MISTStyle.CAT_COLORS[i] for i in 1:4]
    barplot!(axis, task_ids, df.value; dodge=variant_ids, color=variant_ids, colormap=colors, colorrange=(1, 4))
    Legend(figure[1, 2], [PolyElement(color=color) for color in colors],
           ["$tokenizer | $position" for (tokenizer, position) in ABLATION_VARIANTS], "Tokenizer | Position Embedding";
           tellheight=false, tellwidth=true)
    return figure
end

function main(args)
    length(args) >= 3 || error("usage: plots.jl ANALYSIS INPUT_CSV OUTPUT_DIR [LAMBDA_MAX_CSV]")
    analysis, input, output = args[1:3]
    # Validate all source data before producing artifacts.
    data = analysis == "transfer" ? transfer_observations(input) :
           analysis == "gas" ? gas_observations(input) :
           analysis == "ablation" ? ablation_observations(input) :
           analysis == "instability" && length(args) == 4 ? instability_observations(input, args[4]) :
           error("Expected transfer, gas, ablation, or instability (with LAMBDA_MAX_CSV)")
    mkpath(output)
    figure = with_theme(MISTStyle.theme()) do
        analysis == "transfer" ? transfer_figure(data) :
        analysis == "gas" ? gas_figure(data) :
        analysis == "ablation" ? ablation_figure(data) : instability_figure(data...)
    end
    stem = Dict("transfer" => "transfer_adv", "gas" => "mist_paper_gas", "ablation" => "ablation", "instability" => "spiking_lambdas")[analysis]
    MISTStyle.savefig(stem, figure; fig_dir=output)
    if analysis == "instability"
        CSV.write(joinpath(output, "training_loss.csv"), data[1])
        CSV.write(joinpath(output, "lambda_max.csv"), data[2])
    else
        CSV.write(joinpath(output, stem * ".csv"), data)
    end
    return nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main(ARGS)
