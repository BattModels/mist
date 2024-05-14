# Electrolyte Foundation Model
Benchmarking RoBERTa model pre-training on molecular datasets.

# Installation

## Polaris

1. Install [rust](https://www.rust-lang.org/tools/install)

2. Load conda
```shell
module purge
module use /soft/modulefiles/
module --ignore_cache load conda/2024-04-29
module conda/2024-04-25
```

3. Install poetry + pipx
```shell
python -m pip install pipx
python -m pipx ensurepath
python -m pipx install maturin
python -m pipx install --python $(which python) poetry
```

4. Install environment: `poetry install`

5. (optional) Activate `conda` environment: `conda activate base`

## Artemis

Same as above except:
1. Skip loading conda (just use poetry)
2. Ensure a module for CUDA@12.2 exists, may need to install with spack (make sure `buildable: True`)

# Submitting Jobs

```shell
source activate # Activate Environment
./submit/submit.py ./submit/polaris.j2 --data ./submit/pretrain.yaml | qsub
```

See `submit/submit.py --help` for more info

## Hackathon

Create a file `hack.yml` and include it as an overlay to `submit.py` (i.e. `./submit/submit.py ... --data hack.py ...`).
Put the following in `hack.yml`:
```yaml
queue: S1880287
account: GPU_Hack
walltime: "1:0:0"
train:
    data.path: /grand/gpu_hack/FoundEnergy/realspace_v3_dev
```

# Development

## Pre-commit

We use [pre-commit](https://pre-commit.com) to preform various linting checks on the code. To enable:

1. Install poetry (See above)
2. Run pre-commit: `pre-commit`
3. Run before committing: `pre-commit install --allow-missing-config`
