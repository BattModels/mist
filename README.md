# Electrolyte Foundation Model
Benchmarking RoBERTa model pre-training on molecular datasets.

# Installation

The following provides installation instructions for the top-level package (`electrolyte_fm`), optional add-ons for our
various additional analysis and downstream applications (See `opt/`) may require additional configuration.

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

> See `submit/dgx.j2` or `submit/delta.j2` for a more complete example of using the container

# Submitting Jobs

```shell
source ./activate # Activate Environment
./submit/submit.py ./submit/polaris.j2 --data ./submit/pretrain.yaml | qsub
```

See `submit/submit.py --help` for more info

# Development

## Pre-commit

We use [pre-commit](https://pre-commit.com) to preform various linting checks on the code. To enable:

1. Install poetry (See above)
2. Run pre-commit: `uv run pre-commit`
3. Run before committing: `uv run pre-commit install --allow-missing-config`
