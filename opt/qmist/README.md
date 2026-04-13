# QM9 Calculation Replication
Scripts to reproduce the [QM9](https://doi.org/10.1038/sdata.2014.22) calculations from the dataset for arbitrary SMILES strings.

## Installation

Run `./install.sh` from this directory to install the necessary dependencies.

## Usage

See `./main.py --help` for instructions on how to run the script.
Jobs can be submitted to SLURM with `sbatch ./submit.sh ...`

Expected results file `qm9.jsonl` can be generated with `./ref.py generate`
