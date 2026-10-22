# Pretraining supplementary figures

`workflow.pomset` declares the four pretraining supplementary figures and the
unchanged scaling-workflow schematic. Every figure task outputs the PDF, PNG,
and the actual observations used in the plot. `FIGURES` exposes the five PDF
artifacts for paper assembly. These tasks do not train or load models.

| Task | Paper artifact | Source |
| --- | --- | --- |
| `pretraining-transfer` | `transfer_adv.pdf` | Released `mist_pretraining_advantage.csv`, through `scaling-materialize-release` |
| `pretraining-gas` | `mist_paper_gas.pdf` | Original six throughput measurements, supplied below |
| `pretraining-instability` | `spiking_lambdas.pdf` | Original loss and per-matrix eigenvalue traces, supplied below |
| `pretraining-ablation` | `ablation.pdf` | Original selected eight-task/four-variant validation results, supplied below |
| `pretraining-scaling-schematic` | `scaling_workflow.pdf` | Static drawing from the pinned paper source archive |

For example, from the repository root:

```sh
pomset submit --task pretraining-transfer
pomset submit --task 'pretraining-*'
```

The GAS, instability, and ablation tasks deliberately fail with a missing-input
error until their original numeric observations are supplied. Published PDF
assets, digitized points, validation-loss substitutes, and readiness dashboards
are not counted as regenerated experimental figures. Their original plotting
scripts were not found in the checked repository history; the renderers here
implement the panel definitions from the paper and consume explicit tables.
Exact source tables and run selection remain necessary to reproduce those plots.

## Released transfer figure

The transfer task reuses the Bayesian-scaling workflow's pinned released CSV;
it does not download the archive twice. It selects encoder identifiers
`ti624ev1` (28M) and `dh61satt` (1.8B), the sizes shown in the paper, and both
encoder-freeze and pretrained/random policies. The released 228M runs are
excluded. The selected 78 observations form eight curves, with 9 or 10
observations per curve; duplicate size/policy/sample-count cells are rejected.

The plotted loss is the released `val/loss_epoch.min`. Dataset sizes are
`cli.trainer.limit_train_batches * 8`, using the tmQM experiment's batch size of
eight. This yields 6,928–69,328 samples, matching the released figure axis. The
paper prose says approximately 900–70,000; a 900-sample point is not present in
the selected released observations and is not invented here. Preserve the
original training configurations if an independent audit of this conversion is
required.

The scatterline, model-color, and pretrained-circle/random-square conventions
reuse `opt/pretrain_adv.jl` from repository commit
`de0268fb3d57178ef3bbe293e48f063110c58bdf`, adapted to the published two-panel
unfrozen/frozen arrangement. The native CSV is the source of the values; the old
untracked `pretrain_adv.json` is not required.

## Original throughput observations

Supply `data/user/pretraining/gas_throughput.csv` with columns:

```text
run_id,gpus,gas,microbatch_size,samples_per_second
```

Supply exactly one original selected measurement for each pair in
`gpus ∈ {64,256,512}` × `gas ∈ {1,64}`, all with `microbatch_size=512`.
`samples_per_second` is the measured global training throughput, not a
per-device rate or a value inferred from plotted pixels. `run_id` should retain
the original profiler/log identity. If raw samples need averaging, provide the
original averaging window and selection protocol alongside the table; this
renderer does not choose a new averaging rule. The normalized released W&B logs
only establish runs through 64 GPUs and cannot supply the missing larger-scale
measurements.

The published figure legend uses GAS 64, while nearby prose discusses a peak at
GAS eight. This task reproduces the figure's six observations; it does not
silently reinterpret its legend.

## Original instability observations

Supply these two files:

```text
data/user/pretraining/training_loss.csv
run_id,condition,step,train_loss,spike_step

data/user/pretraining/lambda_max.csv
run_id,condition,step,matrix_group,layer,matrix,lambda_max
```

`condition` is exactly `Spiking` or `Corrected`, with one original run identity
per condition shared by both files. Supply actual training loss (not validation
loss). `spike_step` is the common original vertical marker, repeated for each
loss row. Eigenvalue rows retain every original layer/matrix observation used
in the distributions; `matrix_group` is `Feedforward` or `KQV`, `layer` is the
layer index, and `matrix` is its original weight-matrix name. Provide the
original definition/normalization of maximum eigenvalue with these exported
measurements; the task does not derive a substitute weight statistic. A single
mean per checkpoint is insufficient to recover a boxplot.

The renderer creates the two per-step eigenvalue-distribution panels and the
paired training-loss panel. It does not smooth the traces, rerun pretraining,
or select replacement runs. The released normalized validation histories do
not contain these required training-loss and per-matrix eigenvalue series.

## Original architecture/tokenizer ablation

Supply `data/user/pretraining/ablation.csv` with columns:

```text
run_id,dataset,tokenizer,position_embedding,metric,value
```

Provide exactly 32 selected results: each dataset in `bbbp`, `clintox`, `esol`,
`freesolv`, `qm9`, `bace`, `tox21`, `toxcast`, crossed with these four variants:

| tokenizer | position_embedding |
| --- | --- |
| Smirk-GPE | Absolute |
| Smirk-GPE | Rotary |
| Smirk | Absolute |
| Smirk-SELFIES | Rotary |

`metric` is `r2` for ESOL, FreeSolv, and QM9, and `auroc` for the five
classification tasks. `value` is the original validation metric, with QM9's
mean target R² as described by the caption. `run_id` records the exact original
selection; the task does not pick the highest metric from an unspecified sweep.
Retain the original split, seed, checkpoint-selection rule, and any aggregation
rule alongside the table if provenance is to be audited. The exact 32 selected
run/results records have not been identified in the public release.

## Static schematic and verification

The static drawing is extracted from `data/paper-source/2510.18900.tar` only
after SHA-256 verification against
`82e4af73525de5f4a196e36da205edee3e0977c0b1ffb734812247b6b7c014c7`.
The `pretraining-source-archive` task downloads and verifies the archive using
the shared benchmark Python environment.
Only `scaling_workflow.pdf` is extracted; the experimental plots are regenerated
by their own tasks. No model repositories or checkpoints are downloaded by this
module.

Run the input-contract checks with:

```sh
julia --startup-file=no --project=opt/pretraining opt/pretraining/test/runtests.jl
```

Direct rendering uses `plots.jl ANALYSIS INPUT_CSV OUTPUT_DIR`, with the
eigenvalue CSV as a fourth argument for `instability`. The analyses are
`transfer`, `gas`, `instability`, and `ablation`.
