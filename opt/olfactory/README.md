# Olfactory

Scripts for exploring MIST's olfaction model and generating relevant figures from the paper

## Reproducing Analysis

1. Install [julia](https://julialang.org/downloads/) and the base project (See [Project README](../../README.md))
2. Instantiate the environment `uv run julia --project -e 'using Pkg; Pkg.instantiate()'`
3. Obtain model files and place at the appropriate path (see `discordance.jl` and `olfactory.jl`)
4. Run the script: `uv run julia --project discordance.jl`
