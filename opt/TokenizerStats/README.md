# Analysis Code for "Smirk: An Atomically Complete Tokenizer for Molecular Foundation Models"

## Installation

1. First, install [Python v3.12](https://www.python.org/downloads/), [Poetry](https://python-poetry.org), [Rust](https://www.rust-lang.org/tools/install) and [Julia v1.11](https://julialang.org/downloads/).
2. Instantiate the python environment: `poetry install`
3. Activate the environment

```shell
poetry shell
export JULIA_CONDAPKG_BACKEND=Null
export JULIA_PYTHONCALL_EXE="
```

4. Install Julia dependencies: `julia --project -e 'using Pkg; Pkg.instantiate()'`

> If you are collecting tokenizer statistics, see [MPI.jl](https://juliaparallel.org/MPI.jl/stable/) for
> instructions on configuring MPI to use your system (i.e cluster) provided
> MPI installation.


## Reproducing Plots

Complete the installation steps, then:

```shell
julia --project=plots -e 'using Pkg; Pkg.instatiate()'
julia --project=plots -e 'using SmirkPaperPlots; SmirkPaperPlots.main()'
```

## Reproducing Analysis

### Tokenizer Coverage
Complete the installation, then `poetry run python src/atomic_oov.py --help`

### N-Gram Statistics
Complete the installation, then run `julia --project ./main.jl --help`

