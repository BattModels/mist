#!/bin/bash
#SBATCH --job-name=explore
#SBATCH --partition cpuq
#SBATCH -c 4
#SBATCH --mem-per-cpu 4000M
#SBATCH --time 2:0:0

GIT_ROOT=$(git rev-parse --show-toplevel)
MODEL_DIR=../../linear-probes/
source "${GIT_ROOT}/activate"
julia --project --startup-file=no -e 'using Pkg; Pkg.instantiate()'
./explore_probes.jl $@
