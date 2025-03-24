# Electrolyte Foundation Model
Benchmarking RoBERTa model pre-training on molecular datasets.

# Installation

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

# Submitting Jobs

```shell
source ./activate # Activate Environment
./submit/submit.py ./submit/polaris.j2 --data ./submit/pretrain.yaml | qsub
```

See `submit/submit.py --help` for more info

## Building Apptainer Image

```shell
apptainer build --fakeroot \
    --build-arg SSH_AUTH_SOCK=$SSH_AUTH_SOCK \
    mist.sif mist.def
```

## Hackathon

Create a file `hack.yaml` and include it as an overlay to `submit.py` (i.e. `./submit/submit.py ... --data hack.yaml ...`).
Put the following in `hack.yaml`:
```yaml
queue: debug
account: GPU_Hack
nodes: 2
walltime: 1:0:0
train:
  data.path: /grand/gpu_hack/FoundEnergy/realspace_v3_dev
```

# Development

## Pre-commit

We use [pre-commit](https://pre-commit.com) to preform various linting checks on the code. To enable:

1. Install poetry (See above)
2. Run pre-commit: `pre-commit`
3. Run before committing: `pre-commit install --allow-missing-config`
