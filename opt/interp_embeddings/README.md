# Interpretable Embeddings

Scripts for exploring MIST's embeddings and generating relevant figures from the paper

## Reproducing Analysis

1. Install [julia](https://julialang.org/downloads/) and the base project (See [Project README](../../README.md))
2. Instantiate the environment `julia --project -e 'using Pkg; Pkg.instantiate()'`
3. Obtain model files and place at the appropriate path (see `plots.jl`)
4. Run the script: `julia --project plots.jl`
