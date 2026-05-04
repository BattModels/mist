# MIST: Molecular Insight SMILES Transformer

<div align="center" display="flex" >

![GitHub License](https://img.shields.io/github/license/BattModels/mist)
<a href="https://arxiv.org/abs/2510.18900">![arXiv:2409.15370](https://img.shields.io/badge/cs.LG-2409.15370-b31b1b?style=flat&amp;logo=arxiv&amp;logoColor=red)</a>
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/mist-models)

</div>


MIST is a family of molecular foundation models for molecular property prediction.
The models were pre-trained on [Smirk 😏 tokenized](https://github.com/BattModels/smirk) SMILES strings from the [Enamine REAL Space](https://enamine.net/compound-collections/real-compounds/real-space-navigator) dataset using the Masked Language Modeling (MLM) objective, then fine-tuned for downstream prediction tasks.

# Installation

The following provides installation instructions for the top-level package (`electrolyte_fm`), optional add-ons for our
various additional analysis and downstream applications (See [`./opt`](./opt/) may require additional configuration.

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and [julia](https://julialang.org/downloads/) (only needed for `/opt` tasks)
2. Instantiate the environment: `uv sync`
3. Use [`submit/submit.py`](./submit/submit.py) to submit a training job or checkout one of our applications in [`./opt`](./opt)

> You may need to install [rust](https://www.rust-lang.org/tools/install) if pre-built wheels for [smirk](https://github.com/BattModels/smirk) are not available on [PyPI](https://pypi.org/project/smirk/).
> Feel free to [open an issue](https://github.com/BattModels/smirk/issues) to request additional pre-built wheels.

## Polaris

1. Install [rust](https://www.rust-lang.org/tools/install) and [uv](https://docs.astral.sh/uv/getting-started/installation/)

2. Load conda
```shell
module purge
module use /soft/modulefiles/
module --ignore_cache load conda/2024-04-29
conda activate base
```

3. Install the environment
```shell
uv sync
```
## Artemis

Same as above except:
1. Skip loading conda (just use uv)
2. Ensure a module for CUDA@12.2 exists, may need to install with spack (make sure `buildable: True`)

## Apptainer

0. Install or load from a module [Apptainer](https://apptainer.org/)
1. Build the image `bash container/build.sh`, once build relocate the image `mv /tmp/mist.sif ./mist.sif`
2. Run training within the image `apptainer run --nv mist.sif python train.py ...`

> See [`submit/dgx.j2`](./submit/dgx.j2) or [`submit/delta.j2`](./submit/delta.j2) for a more complete example of using the container

# Submitting Jobs

We use a python script ([`submit/submit.py`](./submit/submit.py)) to template training jobs for submission on HPC systems across multiple sites.
Templates may need to be modified for your particular HPC cluster, but should provide a starting point.

```shell
source ./activate # Activate Environment
./submit/submit.py ./submit/polaris.j2 --data ./submit/pretrain.yaml | qsub
```

See `submit/submit.py --help` for more info

> Note: [./activate](./activate) is used to activate the python virtual environment *and* set various environment variables.

# Development

## Pre-commit

We use [pre-commit](https://pre-commit.com) to preform various linting checks on the code. To enable:

1. Install poetry (See above)
2. Run pre-commit: `uv run pre-commit`
3. Run before committing: `uv run pre-commit install --allow-missing-config`
