# BayesianScaling

A Julia Package for fitting regression models using MCMC, that was used to fit penalized neural scaling laws.
To install:

- Install Julia: https://julialang.org/downloads/
- Instantiate the package: `julia --project -e 'using Pkg; Pkg.instantiate()`
- Download the wandb records or chains ([doi:10.5281/zenodo.17527149](https://doi.org/10.5281/zenodo.17527149))

## Code Organization

- `./scripts/` are used for fitting and analyzing the neural scaling laws.
- `./plots/` has plotting code for the paper and various conferences
- `./src` is the MCMC regression and analysis package powering this work
    - [ppl.jl](./src/ppl.jl): Define a regression first interface for fitting MCMC models,
        plus single-pass algorithms for working with posterior samples
    - [scaling.jl](./src/scaling.jl): functional forms for neural scaling laws and derived qualities
    - [analysis.jl](./src/analysis.jl.jl): Code for predicting the perform of models using fitted neural scaling laws
- `./test/` has the unit tests for the BayesianScaling.jl package
- `./benchmark/`: benchmark suite for evaluating different AD backends using [PkgJogger.jl](https://github.com/awadell1/PkgJogger.jl)
