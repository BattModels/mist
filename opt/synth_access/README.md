# Synthetic Accessibility

Evaluating if MIST's molecular surprise correlates with a chemist's ``Synthetic Accessibility''.
Evaluates multiple metrics for synthetic accessibility on two datasets reporting results, plus analysis/plotting code to access the results.

## Workflow

From this directory, inspect with `pomset submit --dry-run`,
then submit with `pomset submit`. Pomset autodetects the backend.
The root `workflow.pomset` also loads this graph through `POMSET_TASKS`.

The workflow stages MIST-28M/MIST-1.8B and the MoLFormer/ChemBERTa baselines
through `opt/benchmarks`. It downloads both accessibility datasets and the
heuristic resources on the head node, then evaluates trained and untrained
models using all three existing encodings and both surprise normalizations.
The heuristic evaluation is a dataset job array.

`synth-access-plots` compares both MIST sizes together with the baselines in
16 PNG/PDF figure pairs and a LaTeX table under `results/synth_access/fig`.
`plot.jl` is an executable Julia `@main` entrypoint and all figures use
`MISTStyle`.

Environment setup needs Rust, libclang, and a C/C++ toolchain for the assembly
library. Its standard-header directory is discovered from the host compiler,
not hard-coded to the development workstation. Molecular assembly has a
per-molecule timeout; unavailable scores remain missing.
Score exports preserve those entries as JSON `null`, never non-standard
`NaN`, and retain their row IDs. The Julia report aligns all scoring tasks by
row ID and rejects missing/duplicate rows or conflicting source labels and
molecules, rather than silently changing the comparison cohort.

The native serialization regression tests run in this folder's environment:

```bash
uv run --frozen --project opt/synth_access --with pytest python -m pytest opt/synth_access/test
```

Run that command from the repository root. The tests use small local molecule
fixtures, including an unavailable assembly score; no model downloads are needed.
