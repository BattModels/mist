# Analysis Code for "Tokenization for Molecular Foundation Models"

<div align="center" display="flex" >

![GitHub License](https://img.shields.io/github/license/BattModels/smirk)
<a href="https://doi.org/10.1021/acs.jcim.5c01856">![paper](https://img.shields.io/badge/paper-10.1021%2Facs.jcim.5c01856-blue)</a>
<a href="https://doi.org/10.5281/zenodo.13761262">![data](https://img.shields.io/badge/data-10.5281%2Fzenodo.13761262-blue)</a>
<a href="https://arxiv.org/abs/2409.15370">![arXiv:2409.15370](https://img.shields.io/badge/cs.LG-2409.15370-b31b1b?style=flat&amp;logo=arxiv&amp;logoColor=red)</a>

</div>

## Installation

1. First, install [Python v3.12](https://www.python.org/downloads/), [uv](https://docs.astral.sh/uv/getting-started/installation/), [Rust](https://www.rust-lang.org/tools/install) and [Julia v1.11](https://julialang.org/downloads/).
2. Instantiate the python environment: `uv sync`
3. Activate the environment: `source ./activate`
> You may need to modify `./activate` for your system

4. Install Julia dependencies: `julia --project -e 'using Pkg; Pkg.instantiate()'`

> If you are collecting tokenizer statistics, see [MPI.jl](https://juliaparallel.org/MPI.jl/stable/) for
> instructions on configuring MPI to use your system (i.e cluster) provided
> MPI installation and update `./activate` accordingly


## Reproducing Plots
Complete the installation then:

1. Activate the environment: `source ./activate`
2. Install the plotting dependencies: `julia --project=plots -e 'using Pkg; Pkg.instantiate()'`
2. Regenerate analysis files or retrieve files from [https://doi.org/10.5281/zenodo.13761263]()
2. Generate plots: `julia --project=plots -e 'using SmirkPaperPlots; SmirkPaperPlots.main'`

### Data Drop Files

The following files should be downloaded from the [drop drop](https://doi.org/10.5281/zenodo.13761263), and placed in the indicated locations.
All commands should be run from this directory.

- N-Gram Statistics (`stats.tar.gz`): `mkdir stats && tar -xvf stats.tar.gz -C ./stats`
- Transformer Models (`models.tar.gz`): `mkdir models && tar -xvf models.tar.gz ./wandb-export -C ./models`

## Reproducing Analysis

### Tokenizer Coverage
Complete the installation, then `poetry run python src/atomic_oov.py --help`

### N-Gram Statistics
Complete the installation, then run `julia --project ./main.jl --help`
