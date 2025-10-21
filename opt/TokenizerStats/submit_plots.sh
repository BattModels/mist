#!/bin/bash
#SBATCH -p venkvis-cpu
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 1800M
#SBATCH --time=0:20:00
set -x
my_job_header

# Move to git root
cd "$(git rev-parse --show-toplevel)/opt/TokenizerStats"

# Activate Environment
source ./activate
env

julia --color=no --startup-file=no --project=plots ./plots/plots.jl
