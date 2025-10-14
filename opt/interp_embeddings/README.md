# Scripts for Exploring MIST's token Embeddings

## Installation

1. Install [julia] and the base environment (See [Project's README](../../README.md))
2. Instantiate the environment: `julia --project -e 'using Pkg; Pkg.instantiate()`

[julia]: https://julialang.org/downloads/

## Generating Plots

See `plots.jl` for the code used to generate plots from the paper. To run you will need
to acquire the pretrained & finetuned MIST models and place at the indicated path (see script).
