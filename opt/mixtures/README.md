# Mixtures

Code for evaluating the MIST mixture models, exploring mixture space and optimizing mixture composition.

## Installation

> All commands run from this directory

1. Install [Julia](https://julialang.org/downloads/) and [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Instantiate the project: `uv run julia --project -e 'using Pkg; Pkg.instantiate()'`
3. Obtain the mixtures dataset from [doi:10.5281/zenodo.17527149](https://doi.org/10.5281/zenodo.17527149)

## Reproducing Plots

Once installed, most of the scripts in the current directory can be run with:
- python: `uv run <script>.py`
- julia: `uv run julia --project <script>.jl`
