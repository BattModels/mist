# Evaluating Chemical Trends with MIST

Source code for querying the MIST models on hydrocarbon and other templatable organic molecules.

## Workflow

Inspect and submit from the repository root with `pomset submit --dry-run`
and `pomset submit`. Pomset autodetects the backend. The root `workflow.pomset`
loads this graph through `POMSET_TASKS`; do not create a separate submission
workspace under this folder.

Both MIST-28M and MIST-1.8B use checkpoints selected/staged by
`opt/benchmarks`, including baseline and randomized QM9 heads. The workflow
runs the existing hydrocarbon/fatty-acid, permutation, olfaction, and
electrolyte analyses separately for each family. It reads the existing local
`opt/design/electrolytes.csv` and generates its atom-level charges with that family's
PubChemQC head; the charge JSONL is a prediction artifact, not an input dataset.

PNG/PDF figures, report JSON, and electrolyte prediction CSVs are written to
`results/design/<base>/<analysis>/`. `plots.jl` is an executable `@main` CLI
using `uv run julia` through its shebang, so PythonCall uses the folder's uv
environment.

Published 28M application heads are consumed directly from their staged
contracts; their held-out evaluations do not block these analyses. Training
missing 1.8B heads and reusing completed heads are handled by `opt/benchmarks`,
which preserves the established datasets and split order (including the cached
KT data and Zenodo pKa data). The DN/BF3-affinity source is still missing;
it blocks only the 1.8B electrolyte-analysis branch. Hydrocarbon, permutation,
olfaction, and partial-charge tasks do not depend on it. Both
families' charge heads require the recovered JCIM2017 Arrow corpus. See
[the benchmark input notes](../benchmarks/README.md). Missing source data is an
explicit failure, not replaced with invented measurements.
