# MIST MoleculeNet benchmarks

`workflow.pomset` evaluates the Cartesian product of the MIST-28M and
MIST-1.8B bases with the 12 MoleculeNet datasets listed in `registry.yaml`.
Dataset targets, metrics, transforms, and split contracts come from
`submit/moleculenet_tasks.libsonnet`; the registry contains only acquisition
policy, immutable model releases, sweep settings, and paper-table presentation.

An exact pinned fine-tuned release is staged for 22 of the 24 model/dataset
pairs. Missing full-QM9 and MUV MIST-1.8B models use the expanded LR arrays
described below. Their test tasks select the
model with the minimum validation loss before it opens the held-out test split. Every
chosen model is scored on test once, and the terminal shell task emits unified
JSON, test CSV, and publication-ready LaTeX/CSV paper tables.

Training GPU allocations are capped at eight hours. Arrays never impose a Pomset concurrency
limit. Downloads run on the head node; GPU tasks use the staged snapshots in
offline mode.

Benchmark commands run as modules from the repository root, using the staged
environment through `uv run --no-sync --offline --project opt/benchmarks -m`.
Pomset declares the environment sync as a prerequisite and sets the working
directory to the repository root.
For example, from the repository root:

```shell
uv run --no-sync --offline --project opt/benchmarks -m opt.benchmarks.scripts.train_screening_head --help
```

The `scripts/train_*.py` wrappers write configurations and invoke the repository's
`python -m train fit` entrypoint with the same interpreter. Mixture, conductivity,
and olfactory-mixture wrappers share configuration and resume handling in
`scripts/training.py`; model-specific
construction and export stay in their respective wrappers. Their explicit
callbacks retain the existing checkpoint paths, validation selection, and job
timer. Resuming uses the supplied configuration for construction and restores
weights, optimizer, progress, and callback state from `last.ckpt`, including
checkpoints from the previous direct-`Trainer` scripts.

## MoleculeNet training-seed uncertainty

Each of the 24 model/dataset pairs also has a five-seed loss report under
`results/benchmarks/mist-validation/<base>/<dataset>/seeds-expanded/losses.json`, exposed
by `MOLECULENET_SEED_LOSSES`. The fixed seeds are 42, 43, 44, 45, and 46;
the canonical train/validation/test memberships stay fixed.

Each pair adds four randomized-SMILES arrays: frozen/unfrozen crossed with AdamW
betas `(0.9, 0.999)` / `(0.87, 0.997)`. They use eight logarithmically spaced rates
per base in `expanded_learning_rates`: `1e-5` to `4e-3` for 28M and `1e-5` to
`1e-3` for 1.8B, including both endpoints. They use `encoding: smiles` and
training-only randomization. Validation uses the supplied deterministic SMILES.
The `-lr8` array names and directories preserve the canceled 15-rate runs' files.
The original seven-rate arrays retain their task identities, items and output
paths. Selection includes all six arrays (46 cells per pair), using validation
loss only, with the smaller LR breaking exact ties. The same combined selection
also includes both beta recipes in other benchmark sweeps.

Expanded arrays retain the 100,000-step ceiling and allow 7h45m of fitting in an
eight-hour allocation, leaving time for selection and export. Batch settings
and existing data splits are unchanged. QM9 remains a joint 12-target fit.
Across 24 pairs this adds 96 arrays / 768 training cells, followed by 96 additional
seed fits and 48 evaluation/collation jobs: 864 training runs / 912 task instances,
excluding completed prerequisites. The seed reports have
new task identities ending in `-expanded`, preserving historical results.
These sweeps keep the usual model artifacts;
the four additional seeds disable all checkpoint writing and model export,
retaining their best validation weights in CPU memory until evaluation finishes.
An interrupted additional-seed task starts fresh.

Every additional seed inherits the complete winning recipe: LR, betas, warmup,
freezing, encoding, randomization, batch sizes, schedule and time budget. A
historical winner retains its original 5h45m fit budget. New sweeps use effective
batch 128 with microbatch 16 for 28M and 1 for 1.8B, except 1.8B MUV (batch 32,
microbatch 4).
Additional seeds initialize from the pretrained base, not the seed-42 winner.
Validation and test losses are evaluated at each seed's best validation state.
For regression this is the existing training-standardized masked MSE; for
classification it is masked binary cross-entropy with logits. Test loss is
never used to select an LR or weights.

The report includes all five individual losses and their mean, sample standard
deviation (`ddof=1`), and standard error. This measures variation across training
seeds on fixed splits, not uncertainty from choosing different data splits.
Per-seed `losses.json` files and CSV training logs remain available. Existing
released model exports and paper-reference benchmark tables retain their paths.
No additional Hugging Face model repositories are needed.

Launch the complete expanded seed chains with:

```shell
pomset submit --task 'benchmark-*-seed-uncertainty-expanded' --wait 10m
```

Selection, the four seed repeats, and final collation use scheduler dependencies.

The workflow also exports application-head model directories for BP, MP, FP, BF3
affinity, odour, DMSO pKa, ETN, and the alpha/beta Kamlet--Taft component. Pinned 28M
releases are reused; 1.8B heads use the same seven-rate validation-only sweep.
Completed BP, MP, odour, DMSO pKa, ETN, and KT runs keep their original datasets,
split membership, and within-split row order. Their paths remain under
`data/screening-heads/replacement-seed42/`; their results must not be relabelled
as evaluations on newly supplied CSVs. In particular, pKa continues to use the
cached Zenodo 16882613 v1.4.0 data, not the newly supplied pKa table.

The existing materializers use the original files in
`data/releases/screening-heads-public/`: BP training/prediction JSONL, the MP
annotated CSV, the pinned OpenPOM CSV, `pKa_Data_Table-v1.4.0.csv` (or its XLSX),
and the saved `stenutz-etn-snapshot.json` / `stenutz-kt-snapshot.json` files.
Their original curation is unchanged. BP/MP retain Python's seeded shuffle;
odour retains pandas' seeded sample; pKa/ETN/KT retain NumPy's seeded permutation
and sorted selected indices. These are intentionally not replaced with the
shared training splitter: the same seed alone does not preserve membership or
order across different algorithms. The shared training helper is unchanged.

Existing split files are checked before any write. A content or row-order
mismatch fails explicitly; matching CSVs and their provenance files are left
untouched, including timestamps. OpenPOM is fetched from its pinned public
revision when absent; the other established source tables must be restored
if missing.

Olfaction figures consume this same OpenPOM seed-42 split through `ODOUR_DATA`.
`ODOUR_MODELS` reuses the locally fitted 1.8B head and adds a fresh 28M LR sweep,
since the released 28M head's original split membership is unknown. The new sweep
uses the existing training entrypoint, seven rates, seed 42, 100,000-step/5h45m
budget, and effective batch 128 (microbatch 16). Its selection task links the
minimum-validation-loss model directory before the olfaction workflow evaluates
held-out rows. Existing application consumers retain their released 28M head.

Both sizes also fit olfactory-mixture heads from all 780 public OlfBoost
observations, split into 624/78/78 rows with seed 42. The tasks reuse
`MixtureSimilarityModel`, select by validation, and publish model directories
for the olfaction figures. The [olfaction README](../olfactory/README.md)
documents the source files, training settings, split limits, and model inventory.
These fits reuse the staged MIST bases and introduce no additional model downloads.

Only the new FP and BF3-affinity heads use supplied CSVs. Put these files in
`data/benchmarks/sources/` at the repository root:

| Dataset | File | Required columns |
| --- | --- | --- |
| FP | `fp_db_filtered.csv` | `iso_smiles`, `fp` |
| BF3 affinity | `dn.csv` | `smiles`, `BF3 affinity` (kJ/mol) |

Canonical `smiles` column names also work. No files in this input directory
belong in Git; `data/` is ignored. Copy the inputs separately when deploying to
another machine. Put the lithium-air reference CSV at the separately ignored
`opt/screening/lithium_air_reference_solvents.csv`.

New FP/DN CSVs use `electrolyte_fm.data_modules.utils.train_val_test_split`
without reordering, deduplicating, averaging measurements, or filtering rows
before splitting. Encoding/invalid-SMILES filtering remains downstream, matching
the original CSV datamodule's order of operations. Each `protocol.json` records
the source path, column renames, and split counts; matching the algorithm alone
does not establish exact historical curation. New FP/DN splits go under
`data/screening-heads/supplied-seed42/`, separate from the established splits.

The final report requires all 46 model/dataset/variant results, including the screening
heads. `benchmark-tests.tex` and `benchmark-tests.csv` contain the computed
held-out scores; the separately named MoleculeNet paper tables retain their
explicit reference provenance.

## Additional design and probe prerequisites

Baseline/randomized QM9 variants, MUV, and PubChemQC charge heads support the
design and FeatureMiner workflows. PubChemQC heads use four atom-token channels
(Mulliken and Löwdin central-atom charges plus mean attached-hydrogen charges),
unfrozen LR arrays, validation selection, and a following held-out evaluation.
The recovered corpus must exist at
`data/pubchemqc/pubchemqc_jcim2017-split-v2`; its materializer validates and
publishes a directory symlink to the existing Arrow splits without copying or
redownloading 8.7 GB. Fitting and evaluation consume this declared `dataset`
artifact directly, so file exports include the corpus. Separate provenance JSON
records the release and relative split filenames; inference does not read paths
from that JSON.
Later PM6-based PubChemQC releases are not substitutes for JCIM2017.

The isotope branch adds new 28M and 1.8B fits on the available 924-row Wolfram
cohort. `benchmark-isotopes-materialize` reuses the original parser and seed42
80/10/10 row split (739/92/93). Both bases use the existing seven-rate sweep,
5,000 steps per cell and effective batch16; selected model directories are
exported in `ISOTOPE_MODELS`. The original `half_life` coefficient and `log_time`
exponent remain separate standardized targets. Duplicate isotope identities are
preserved, so validation/test are held-out rows. See `opt/isotopes/README.md` for
the explicit curation, training and projection settings. These tasks require no
historical isotope checkpoint or additional Hugging Face repository.

The supplied source bundle adds FP and lithium-air references. Its BP, MP, DMSO
pKa, ETN, and name-only KT tables are retained on disk but do not replace the
established datasets. Existing KT runs continue using the cached snapshot with
SMILES. `dn.csv` still needs to be supplied from Laurence and Gal's BF3-affinity
table; the different Gutmann DN scale is not a substitute. That materialization
fails explicitly until its input is present. Data preparation and smoke tests
do not constitute reproduced paper scores.

Application workflows can consume a pinned released checkpoint directly from
its staging task, just like MoleculeNet releases. They do not wait for its
test-set evaluation. This allows released heads to be used even while their
evaluation datasets are unavailable. All evaluation tasks remain in the
graph, and the final benchmark report still requires every test result. New
LR-sweep heads still depend on the validation selection in their test task;
missing or incomplete training datasets are not bypassed.

Model outputs are loadable directories. Staging publishes a symlink to the pinned
Hugging Face snapshot; fitted heads save their weights, configuration, and
tokenizer together. The workflow exports `BASE_MODELS`, `MOLECULENET_MODELS`,
`APPLICATION_MODELS`, `DESIGN_QM9_MODELS`, and `PUBCHEMQC_MODELS` for downstream
use. Selection JSON records validation results; consumers load the declared
model output directly. `pomset export --with-files` therefore bundles model
contents, including files reached through snapshot and blob symlinks.

## Strict mixture fits

`benchmark-{mist-28m,mist-1p8b}-mixtures-strict-{0..4}` fits each of the five published
`k-compound-strict-*` folds from the pinned HF excess-properties database.
The existing `ExcessPhysicsModel`, loss, normalization, and
`ProgressiveThawing` callback are reused. Each run starts from the staged
corresponding pretrained base, with the released mixture model's architecture
(arrhenius temperature dependence, three control points, difference
interactions, and its configured dropout). No random-split fine-tuned weights
enter a strict fit.

The existing excess-model entrypoint supplies the optimization defaults:
batch 16, AdamW at 5e-5, beta2 cosine warmup, 10,000 steps, float32, and
progressive thawing of the embeddings followed by the last two encoder layers.
The 1.8B fits use microbatch 1, accumulation 16, validation batch 4, and
bfloat16 mixed precision, retaining effective batch 16. The workflow fixes
seed 42 and evaluates the complete validation fold every 100 steps.
Normalization is fitted on training rows only. Checkpoints and
optimizer state are retained across wall-time retries; an incomplete fit is
not published as a completed model. The minimum validation-loss checkpoint is
exported with its tokenizer to `mixtures-strict/fold-N/model`, alongside
`training.json` recording its fold, initialization, parameters, and score.

Each fold has an experimental plot in `opt/mixtures/workflow.pomset`. Fold 0
owns the established task name for 28M, chosen in advance. These are new fits,
not claims of identical historical paper checkpoints. No extra HF model
repository is downloaded: the existing MIST bases and released mixture
architecture are reused.

`benchmark-mist-1p8b-mixtures-random` uses the same trainer on the published
random train/validation split for the 1.8B parity/outlier figures. Selection
and normalization use validation and training, respectively; fitting does not
evaluate the test split.

The additional 1.8B ToxCast head for token interpretation uses the existing
canonical dataset materializer, LR sweep, and model selection. The requested
24-pair MoleculeNet five-seed grid remains unchanged.

`benchmark-mist-1p8b-ionic-conductivity-fit` trains a fresh head from the staged
MIST-1.8B base using the existing prepared AEM seed-42 split. Its materializer
reproduces the split from the pinned public AEM table: hash each transformed
row with seed 42, assigning buckets 0/1 to test/validation and 2–9 to training.
This is the accepted replacement split, not the original paper split.
The fit retains the prepared protocol: unfrozen encoder, AdamW at 3e-4,
batch 1, validation batch 4, DeepSpeed, bfloat16 mixed precision, and 100,000 steps,
with validation every 1,000 steps. It resumes optimizer/checkpoint state after
job limits and uses activation checkpointing and CPU optimizer offload for
40 GB GPUs. It exports the minimum-validation-loss model and tokenizer only
when the full training budget is complete. No test evaluation or additional
HF model repository is required. The old partial checkpoint is not an input.
