# Feature Miner

Code for evaluating fitted linear probes for their ability to predict various chemically meaningful features.

## Workflow

From this directory, inspect both families with:

```sh
pomset submit --dry-run --task feature-miner-mist-28m-plots --task feature-miner-mist-1p8b-plots
```

Remove `--dry-run` to submit. Pomset autodetects the backend. Explicit targets
include only the required benchmark acquisitions; an untargeted submission
also includes the imported benchmark graph.
The root `workflow.pomset` also loads this graph through `POMSET_TASKS`.

The workflow downloads the paper's Lipinski data release and obtains both
MIST-28M and MIST-1.8B encoders through `opt/benchmarks`. For each family it
fits the existing linear probes to the pretrained encoder and seven
task-finetuned encoders. These encoders are deliberately frozen: only the
probes train. Each probe retains its best validation checkpoint.
Training runs `submit_linear_probes.py`, which owns the fit configuration,
resolves benchmark receipts, and invokes the existing probe trainer through
`MistLightningCLI`. It does not load `submit/linear_probe.jsonnet` or define a
separate CLI subclass. Probe dimensions are inferred from the encoder.
The existing Julia checkpoint loader reads the selected probes directly and
saves their analysis as `probes.jld2`; there is no JSON weight export. Plotting
also accepts existing JLD2 probe tables.
The paper's probe budget is 10,000 optimizer steps with constant-learning-rate
AdamW at 0.001, with no epoch limit. A 7.5-hour safety timer keeps the job within
its allocation. Checkpoints record exact validation metrics and selected steps;
a small `fit.pt` records the completed steps and requested budget. Both are
included in the JLD2 table, so a timer-limited fit is not mistaken for a
full-budget replication.

Results are written to `results/FeatureMiner/<base>/`: probe checkpoints,
the existing additivity/feature-alignment analysis, five PNG/PDF figure pairs,
a LaTeX regression table, and the combined JLD2 data.

The workflow installs its own `.venv`; PythonCall plotting uses an executable
Julia entrypoint whose shebang runs `uv run julia`. The older interactive
exploration code remains available separately.
