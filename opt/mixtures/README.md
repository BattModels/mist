# Mixtures

Code for evaluating the MIST mixture models, exploring mixture space and optimizing mixture composition.

## Installation

> All commands run from this directory

1. Install [Julia](https://julialang.org/downloads/) and [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Instantiate the project: `uv run julia --project -e 'using Pkg; Pkg.instantiate()'`
3. Run the workflow's dataset staging tasks to obtain the pinned Hugging Face database and supplementary experimental tables.

## Reproducing Plots

Once installed, most of the scripts in the current directory can be run with:
- python: `uv run <script>.py`
- julia: `uv run julia --project <script>.jl`

## Workflow

`workflow.pomset` exposes the existing dataset, SOAP/ReMATCH, excess-property
parity, composition-extrema, experimental-repeat, and thermal-expansion plots.
`plots.jl` supplies paths to the original plotting functions; their scientific
definitions are unchanged. Tasks declare the PDF/PNG files and, for parity,
the predictions and composition-extrema tables. `FIGURES` exports the PDF
artifacts for figure assembly. Figure outputs are grouped under
`results/mixtures/{mist-28m,mist-1p8b}/` by analysis or strict fold, without a `paper`
directory. Shared database tables and environments remain under
`results/mixtures/`.

The authoritative database for this workflow is the user-accepted
[Hugging Face excess-properties dataset](https://huggingface.co/datasets/mist-models/excess-properties/tree/a42344b3057341a04ce6f2cf76f6f2882fda4e6c),
pinned to `a42344b3057341a04ce6f2cf76f6f2882fda4e6c`.
`mixtures-stage-database` publishes its actual snapshot directory at
`data/mixtures-workflow/excess-properties`. It includes the published `random`,
five `k-compound-*`, and five `k-compound-strict-*` configurations.
The random configuration has 83,279 training, 10,410 validation, and 10,410 test
rows. The cross-validation configurations contain train and validation splits.
Their missing test directories are not errors or invitations to invent splits.

`mixtures-export-database-table` concatenates the three random splits, in
train/validation/test order, into
`results/mixtures/database/excess-properties.csv`. Its 104,099 rows retain all
columns, duplicates, and missing property values. Other configurations reuse
observations and therefore are not concatenated into the full-table export.
`mixtures-paper-dataset` feeds this actual table to the unchanged native dataset
plot. No separately supplied v8p2 CSV or split directory is required.

The published
[MIST excess-model card](https://huggingface.co/mist-models/mist-mixtures-zffffbex/blob/cb54e41fa6c477e705419232752971e4463cd699/README.md)
identifies its evaluation data as an 80/10/10 random split. The workflow uses
that documented released model for `mixtures-paper-parity` and
`mixtures-paper-outliers`, rather than requiring an unidentified `random-model`
directory. The parity task deliberately preserves the original code's
`k-compound-0` **validation** evaluation. Its overlap with the released model's
training membership is unverified, so the resulting parity/extrema plots are
retrospective visualizations, not sealed holdout evidence. Neither the accepted
database nor the model card establishes identical archived figure numerics.

SOAP/ReMATCH and experimental-repeat observations continue to come from the
checksum-pinned Zenodo 21302713 `mixtures.tar.xz` archive. Historical
`mixtures-released-*` v6 views remain available separately. The HF database's 92
Arrow shard hashes match those in that release; this correspondence does not
change the selected HF source or imply a new data curation step.

Strict models are fitted by `benchmark-{mist-28m,mist-1p8b}-mixtures-strict-{0..4}` in
`opt/benchmarks/workflow.pomset`. Each task uses its released
`k-compound-strict-*` train/validation directories, starts from the shared
corresponding pretrained base, and exports the checkpoint with the lowest validation
loss as a standalone model directory. The released random-split weights are
not used to initialize these fits. Its architecture is reused. Training and
normalization use only each fold's training observations.

The 1.8B random-split head is fitted separately by
`benchmark-mist-1p8b-mixtures-random` using the released `random` configuration.
It supplies the 1.8B parity and outlier plots; the released 28M head continues
to supply the 28M plots. The existing parity cohort remains `k-compound-0`
validation for both sizes, so these plots are retrospective visualizations.
`FIGURES_BY_MODEL` exposes both sets; `FIGURES` keeps the original 28M keys.

All five folds of each model produce experimental-repeat plots. The established
`mixtures-paper-experimental` path uses fold 0; the remaining folds are named
`mixtures-strict-fold-{1..4}-experimental`. This choice is fixed before fitting;
experimental measurements do not select a fold. These are newly fitted models,
not recovered historical checkpoints. See `opt/benchmarks/README.md` for the
recorded training protocol.

The following optional measurements remain:

| Path from repository root | Contents |
| --- | --- |
| `data/user/mixtures/alpha_data.jsonc` | Original thermal-expansion measurements used by `plot_thermal_alpha!`. |

Missing inputs produce errors naming the required paths. The random-split
checkpoint is not relabelled as strict. Thermal expansion is an optional
analysis and is not required by the 133-figure bundle.

The full CSV requires `smi1`, `smi2`, `x1`, `doi`,
`temperature [kelvin]`, `pressure [megapascal]`, and the three target columns
`density [gram / centimeter ** 3]`, `molar volume [centimeter ** 3 / mole]`,
`molar enthalpy [joule / mole]`, plus each target prefixed with `excess `.
The Arrow shards require `smi1`, `smi2`, `x1`, `x2`,
`temperature [kelvin]`, and those six target columns; missing targets are null.
The materializer preserves the published rows and split membership.

Each model directory must contain `config.json` and `model.safetensors`.
Both the legacy `ExcessPhysicsModel` config (embedded `encoder` config and
`target_columns`) and the released production `model_type: mist_excess_physics`
config are accepted. Weights must include the encoder, interaction/property
heads, polynomial, and fitted total/excess normalization buffers. The original
evaluation functions use the unmodified Smirk vocabulary built into the
package; supplied strict models must use that vocabulary. A strict checkpoint
identifier and its fold/training protocol are needed to establish that model's
evaluation lineage; naming a directory does not establish its split mapping.

`alpha_data.jsonc` is an array of records with `mixture`, `x1`, and `alpha`.
The mixture keys are exactly `ACN+1,2-ED`, `ACN+PEG`, and `ACN+1,3-PD`;
each `x1`/`alpha` pair contains equal-length arrays of mole fractions and
measured excess thermal expansion in K⁻¹. Line comments beginning with `//`
are accepted by the original parser.

The only new Hugging Face model download is
`mist-models/mist-mixtures-zffffbex` at
`cb54e41fa6c477e705419232752971e4463cd699`, in
`mixtures-stage-released-model`. Its output is a model-directory symlink.
The encoder config and weights are part of that snapshot, and the released
model's Smirk tokenizer ships in the `smirk` package; there is no nested model
or tokenizer download. Compute tasks use the shared offline environment.
The dataset download is separate: repository `mist-models/excess-properties`,
type `dataset`, revision `a42344b3057341a04ce6f2cf76f6f2882fda4e6c`.
Fitting additionally reuses the staged bases: `mist-models/mist-28M-ti624ev1`
at `bfeebb7a2619a61d321de922b66376f5cb682eef` and
`mist-models/mist-1.8B-dh61satt` at
`ed6df0875e4425ede9b12019bbd8c2a3c439052d`. There is no new model repository
or implicit model download; trained strict models are local output artifacts.

`python -m pytest -q opt/mixtures/test/test_dataset_artifacts.py` checks direct
dataset-directory staging, preservation of full-table rows/nulls, and actual
two-split Arrow evaluation with tiny local data. It downloads or runs no model.

## Ionic-conductivity workflow

`ionic_workflow.pomset` reuses the mixture environment and exposes the existing
conductivity plots through `ionic_plots.jl`. It covers the main composition-curve
panel, non-Arrhenius validation, composition curves, all 32 Ea/T0 and excess
ternary surfaces, salt comparisons, and the five molarity-specific descriptor
fits. Descriptor scatter panels use the same model-derived T0 values and
solvent constants as the existing fits. The molarity tasks also save
`unary-properties.csv` and `descriptor-coefficients.csv`. `FIGURES_BY_MODEL` exports both sets of
PDF artifacts under `results/ionic-conductivity/{mist-28m,mist-1p8b}/`;
`delta_Ea` and `soap_corr_T_298` are the components of the
paper's `Ea_comparison_panel` layout.

`ionic-stage-released-model` stages
`mist-models/mist-conductivity-27.0M-2mpg8dcd` at
`61d2c23d7313f14d26a6890fca37cd5f9c4f180b`. This is the checkpoint named in the
original Julia plotting code. Its model directory includes the encoder,
conductivity head, and tokenizer; no nested model/tokenizer download occurs.
The 1.8B figures use the fresh head trained by
`benchmark-mist-1p8b-ionic-conductivity-fit` on the prepared AEM seed-42 split.
The task starts from the pretrained MIST-1.8B base and runs the existing
100,000-step protocol, then exports the validation-selected model directory.
It does not depend on the old partial local checkpoint. See the benchmark
README for the split and fitting parameters.

Each model's `ionic-{base}-prediction-tables` task generates the main-panel and
Angell CSV artifacts under `results/ionic-conductivity/{base}/predictions/`.
The figure tasks depend on those artifacts; neither plot needs a saved table
under `data/user`. The AEM observations and SOAP similarity data are shared.
The same release is locally archived in
`data/releases/zenodo-21302713/mist-mixtures.tar.gz`.
An offline 1M descriptor smoke run reproduces the numerical example in the
paper at its reported precision: LiTFSI coefficients 206.3, 0.388, and -0.095,
with R² 0.989; LiPF6 R² 0.927. This corroborates the checkpoint named in the
plotting code. These entrypoints retain the existing VFT parameter,
conductivity, salt-fraction, and extrapolation calculations; the full set of
plots still requires execution.

The generated tables use these explicit grids, rather than attempting to
recover the paper's original sampled rows:

| Artifact | Columns, units, and evaluation grid |
| --- | --- |
| `ionic_conductivity_curves.csv` | `salt_name`, `temperature`, `composition`, `predictions`, `Solvent`: LiTFSI and LiPF6 in PC at 260, 280, 300, 320, and 340 K, with 50 equally spaced salt mole fractions from 0.02 to 0.20 (500 rows). `predictions` is conductivity in mS/cm, obtained by exponentiating the model's log conductivity. The existing composition-curve helper supplies the evaluation; EC/FEC occupy its zero-weight solvent slots. |
| `ionic_conductivity_inference.csv` | `Solvent`, `salt_name`, `molarity`, `temperature`, `composition`, `Tg`, `Tg/T`, `conductivity`: LiPF6 at 1 M in each of FEC, PC, EMC, EC, DEC, and DMC, evaluated every 5 K from 260 through 340 K (102 rows). The existing density/molecular-weight conversion sets each solvent's salt mole fraction. `conductivity` remains natural-log conductivity in mS/cm. `Tg` is the model's historical name for the VFT **T0** parameter in K; `Tg/T` divides this prediction by temperature and is not an experimental glass-transition measurement. |

Rows are ordered by salt, temperature, and composition for the main curves,
and by solvent and temperature for Angell. The main renderer uses every
generated row and scales its vertical range to the predictions. Both plots
retain the MIST theme. No additional model is downloaded for these tables;
they use the same staged 28M or freshly fitted 1.8B model as the other ionic
figures.

The non-Arrhenius task consumes
`data/external/DiffMix-NatCommData/training_data/published_AEM_data.csv`, which
is present locally. Its public source is the `BattModels/DiffMix-NatCommData`
GitHub repository at commit `a3651ba671d2e54ed3a16f27d7d334a3339b7f54`; the file
SHA-256 is `2f44ba6e69c07bbee4c914afa5b47ad9ae2b0769eb951b22330f4ecf3a244c7f`.
It requires `sub1_name`, `sub2_name`, `sub3_name`, `salt`,
`x1 (mole fraction)`, `x2 (mole fraction)`, `x3 (mole fraction)`,
`x4 (anion, mole fraction)`, `x5 (cation, mole fraction)`, `temperature (K)`,
and `ionic conductivity (mS/cm)`. Preserve the original row order: the existing
plot chooses the first qualifying mixture groups after its high-salt filter.
`solvent_rematch.json` is available in the released mixture archive and maps
SMILES-pair keys of the form `smiles1_smiles2` to ReMATCH similarities.
