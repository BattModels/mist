# High-Throughput Screening with MIST

The [Pomset workflow](./workflow.pomset) runs electrolyte and lithium-air screening
with the 28M and 1.8B models from [benchmarks](../benchmarks/workflow.pomset).
Separate sweeps collect scaling measurements.

Provide `opt/screening/lithium_air_reference_solvents.csv` with a `smi` column.
From the repository root:

```bash
pomset submit --task screening-generate-performance-sweep
pomset submit --task 'screening-*'
```

The generator writes `campaigns/performance.json` as `{"runs": [...]}` with
unnamed parameter sets. Generate it first, then submit again so Pomset can create
the tasks and assign their names. To run one production campaign:

```bash
pomset submit --task screening-mist-28m-electrolyte
```

The workflow writes each campaign configuration using the declared benchmark
model directories, then launches `screen.py` directly with `srun`.

Outputs are under `results/screening/<family>/`. Use each script's `--help` for
its arguments.

Lithium-air figures and structure pages use the current production campaign.
The workflow does not require the paper's historical screening database or
candidate membership.

## Sample Screening Campaigns

- [Electrolytes](./config.yaml)
- [Fragrance, Non-Toxic Liquids](./config_fragrance.yaml)
- [Tom Ford's Eau de Soleil Fragrance](./config_tf_eau_de_soleil.yaml)
- [Tom Ford's Oud Wood Fragrance](./config_tf_oud_wood.yaml)

### Remaining original supplementary inputs

`screening-mist-28m-mixed-halogens-plots` consumes
`data/user/screening/mixed-halogen-descriptors.csv`. Supply the original
observation table with columns `parent_id`, `halogen_type`, `homo`, `lumo`,
`gap`, `bp`, `mp`, `fp`, and `Cv`. `F` identifies the fully fluorinated baseline;
other halogen labels must preserve the paper's substitution categories. Values
must retain the original units (frontier energies/gap in eV, temperatures in
°C, heat capacity in the original reported units). The electronic descriptors
come from the paper's quantum calculations; SMILES alone or new MIST predictions
are insufficient for that validation. The task fits the paper's categorical
halogen-plus-parent fixed-effects model, exports all fitted halogen coefficients
and 95% intervals, and plots effects with p < 0.05. Original molecule membership,
category coding, and descriptor calculations are required for exact parity.

`screening-mist-28m-lithium-air-calibration-plots` consumes
`data/user/screening/lithium-air-calibration.csv`, with columns `solvent`, `beta`,
`ETN`, `Li_vs_SHE`, and `O2_vs_SHE`. Supply the original literature-derived
calibration observations and potentials in volts relative to SHE. Blank cells
are allowed when a solvent was measured for only one couple; each fit needs at
least three complete descriptor/potential pairs. The plot reuses the implemented
`f_SHE_Li` and `f_O2_O2minus` curves without refitting their coefficients.

Both supplementary tasks fail with the missing input path when original data is
absent. They do not replace the validation observations with model predictions.
Released screening replay now exposes every quantum-validation and odor figure,
PDF/PNG pairs, Pareto tables, structure lists, and the calculated Rényi statistics.
These are available through the workflow's `FIGURES` export.

Offline plot checks (no model downloads or model inference):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
JULIA_DEPOT_PATH="$PWD/.cache/julia:" \
JULIA_PYTHONCALL_EXE="$PWD/opt/screening/.venv/bin/python" \
julia --startup-file=no --project=opt/screening/plots \
  opt/screening/plots/test/runtests.jl
```
