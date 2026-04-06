# Feature Miner

Code for evaluating fitted linear probes for their ability to predict various chemically meaningful features.

# Replication

1. Install [Julia](https://julialang.org/downloads/)
2. Instantiate the project: `julia --project -e 'using Pkg; Pkg.instantiate()'`
3. Train linear probes using [linear_probe.jsonnet](../../submit/linear_probe.jsonnet) and [submit/submit.py](../../submit/submit.py) on
MIST finetuned models.
4. Run `julia --project explore_probes.jl` to extract fitted probe weights from the checkpoints
5. Instantiate the plotting code: `julia --project=plots -e 'using Pkg; Pkg.instantiate()'`
6. Evaluate fitted probes: `julia --project=plots ./plots/lipinski_probes.jl`
