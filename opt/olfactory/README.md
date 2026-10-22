# Olfactory

Scripts for exploring MIST's olfaction model and generating relevant figures from the paper

`workflow.pomset` produces odour panels under
`results/olfaction/{mist-28m,mist-1p8b}/`. It reuses the original Julia overview, radial
dendrogram, and discordance plotting functions, and the existing
`MixtureSimilarityModel` for mixture inference. Its `FIGURES` export contains
the individual main-panel PDFs and supplementary figures for final assembly.
Homologous-series and branched-scent plots remain in `opt/design/workflow.pomset`.

The workflow loads each model's validation-selected benchmark **odour head**, generates the
descriptor logits from the public 4,983-molecule OpenPOM release, and produces
the discontinuity examples. Training and held-out figure tables use the same
seed-42 80/10/10 benchmark split. Both heads are fitted on that split; the 28M
analysis does not reuse a released head with unknown training membership.
The clustering command explicitly selects the
paper's `sqrt(2*(1-rho))` distance and average linkage; running
`scent_dendrogram.jl` by itself preserves its historical `1-rho`/complete-linkage
defaults. These settings differ in the original script and paper methods.

`FIGURES_BY_MODEL` exposes both model families. `FIGURES` retains the original
28M assembly keys. Public inputs, reproducible splits, and identity mapping are
shared. The 1.8B geometry tables have a separate input path under
`data/user/olfaction/mist-1p8b/geometry` with the same schemas as the 28M
counterparts below. Geometry and prospective-identity inputs remain on hold.

The benchmark Python environment is reused. `olfaction-julia-environment`
instantiates this Julia project before plotting. All model inference is offline;
the shared benchmark staging tasks are the only model download steps. No
placeholder or readiness PDFs are produced when inputs are missing.

The shared benchmark OpenPOM fetch and three `olfaction-fetch-*` tasks acquire
the public OpenPOM CSV and Osmo `Data S5.csv`, `Data S7.csv`, and `triplets.csv`
tables from immutable GitHub commits with pinned SHA-256 and byte counts.
These public inputs are automated; they do not need to be supplied manually.
Cached matching files are reused.

### Inputs still needed

All paths below are relative to the repository root. These files are consumed
directly by the workflow; a receipt pointing at them is not sufficient.

| Path | Required content |
| --- | --- |
| `data/user/o2-redjade-mapping.csv` | Missing prospective identities: `redjade_code,canonical_isomeric_smiles,source_or_release_id`. Combined with public `Data S7.csv`, this must cover all 46 codes in `triplets.csv`. Current public mapping covers 12; 34 are missing. The existing public ratings and GNN prediction tables are consumed without downloading/running the GNN. |
| `data/user/olfaction/geometry/{euclidean,hyperbolic}_betti.csv` | Ordered `radius` and paired **empirical and best-fit model** curves: `empirical_betti_1`, `empirical_betti_2`, `empirical_betti_3`, `model_betti_1`, `model_betti_2`, `model_betti_3`. |
| `data/user/olfaction/geometry/hydra_coordinates.csv` | Original `molecule_id,dimension_0,dimension_1,dimension_2` projection coordinates. The sphere view uses x/y position, near/far markers by z, and RGB proportional to x/y/z as described in the paper caption. |
| `data/user/olfaction/geometry/molecule_ids.csv` | One `molecule_id` column in the exact HYDRA row order. |

The historical `replication/intake_geometry_bundle.py` archive schema contains
only one set of beta-0/1/2 curves. That is insufficient for the paper's
empirical-versus-model beta-1/2/3 comparison, so this workflow requires the
paired numeric curves explicitly. Supplying original ALBATROSS files may
require lossless conversion to the table format above; fitting new geometry
would be a new experiment.

### Olfactory-mixture fitting

`benchmark-olfactory-mixtures-fetch-*` fetches the unaugmented OlfBoost annotations,
component definitions, and CID-to-SMILES table from commit
`d1779fc9c632e279aed80ae2c945031a8dbf6080`, with pinned SHA-256 and byte counts.
The materialization task resolves all 780 annotations, including the 100
Ravia 4/5 observations previously excluded because zero CID padding was
mistaken for missing identities. Missing nonzero CIDs fail explicitly.

`benchmark-olfactory-mixtures-materialize-split` writes
`data/olfactory-mixtures/seed42/{train,val,test}.csv`: 624/78/78 observations,
JSON-list `mix1_smiles,mix2_smiles`, observed `value`, and source identifiers.
This is a fresh random pair split, not an entity-disjoint split or a claim to
recover the paper's exact membership. All observations are retained once.

`benchmark-{mist-28m,mist-1p8b}-olfactory-mixtures-fit` uses the existing
`MixtureSimilarityModel` (`WeightedSumPooling` and `ScaledCosineDistance`).
The baseline follows `submit/chemixhub.jsonnet`: unfrozen encoder, batch size 2,
validation batch size 4, AdamW at 1.6e-4, and 100,000 cosine-scheduled steps.
An identity normalizer preserves the measured [0,1] targets for the bounded
distance head; no scientific model implementation is changed. The 1.8B fit
uses bf16, activation recomputation, and optimizer CPU offload for memory.
Validation chooses the checkpoint; these fitting tasks do not evaluate the
test split. JobTimer and `last.ckpt` support resuming across allocations.

Each fit publishes `results/benchmarks/mist-validation/BASE/olfactory-mixtures/model/`
containing the selected `model.ckpt`, including head/encoder weights and loading
configuration. The figure workflow consumes that model directory and the
generated split directly. The `olfaction-*-mixture-predictions` and
`olfaction-*-mixture-performance` tasks no longer need supplied checkpoints or
split files. These tasks are separate from the composed overview: published
main panel 3d contains the mixture architecture drawing, not a parity plot.

The overview preparation uses standard held-out per-descriptor and
per-molecule AUROC, and preserves the existing functional-group plotting
definitions. Its functional-group counts therefore count evaluable held-out
molecules, as the plotting code does; the paper caption calls these training
counts. Original intermediate tables or clarification of that discrepancy are
needed to establish exact numerical/layout identity for those panels.

### Model downloads during a real execution

| Task/source | Hugging Face repository | Pinned revision | Components |
| --- | --- | --- | --- |
| Shared `benchmark-stage-mist-28m` | `mist-models/mist-28M-ti624ev1` | `bfeebb7a2619a61d321de922b66376f5cb682eef` | Base encoder and tokenizer for fresh single-molecule and mixture heads. |
| Shared `benchmark-stage-mist-1p8b` | `mist-models/mist-1.8B-dh61satt` | `ed6df0875e4425ede9b12019bbd8c2a3c439052d` | Base encoder and tokenizer for fresh single-molecule and mixture heads. |

No other model repository is introduced by this workflow. GNN profiles are static public tables.
Loading a snapshot uses local files only; missing assets fail instead of
falling back to another Hub model.

### Upgrading existing figure workspaces

The fresh 28M odour head changes the inputs of existing task IDs. Current
Pomset can retain successful outputs after a command changes if those outputs
still validate. Archive any existing artifacts of the following tasks on each
site before resubmitting, and revalidate/retry those tasks as needed:

- `olfaction-all-descriptors-predictions`, `olfaction-held-out-predictions`
- `olfaction-performance-tables`, `olfaction-discontinuity-predictions`
- `olfaction-descriptor-clusters`, `olfaction-overview`, `olfaction-discordance`

`pomset task show TASK --json` lists each task's exact output paths. Revalidating
the predictions and plots prevents syncing older accepted artifacts back into
the updated workspace. Fresh workspaces need no migration.

### Local checks

`opt/benchmarks/.venv/bin/python -m pytest -q opt/olfactory/test_workflow.py
opt/benchmarks/scripts/test_olfactory_mixtures.py` checks held-out alignment,
missing identities, zero padding, deterministic complete splits, workflow model
dependencies, and a two-step fit/reload using a tiny locally constructed encoder.
It downloads no model. `pomset task show 'olfaction-*' --workflow
opt/olfactory/workflow.pomset --json` evaluates task/output dependencies without
executing the analyses.
