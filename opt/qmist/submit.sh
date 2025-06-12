#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 1800M
#SBATCH --time 8:0:0

# Training script for h001
my_job_header
set -ex

module purge
module --ignore_cache load python/3.11.5
module --ignore_cache load Chemistry
module --ignore_cache load gaussian/09-revD01

# Add MOPAC to path
export PATH="$(realpath vendor/mopac*/bin):$PATH"

# Run the pipeline
uv run python ./main.py $@

echo "done: $(date)"
