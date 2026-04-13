# Evaluating Chemical Trends with MIST

Source code for querying the MIST models on hydrocarbon and other templatable organic molecules.

## Installation

> All commands run from this directory

1. Install [Julia](https://julialang.org/downloads/) and [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Instantiate the project: `uv run julia --project -e 'using Pkg; Pkg.instantiate()'`
3. Download the mist models to `models/`
4. Recreate the plots `uv run julia --project plots.jl`
