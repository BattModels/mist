#!/bin/bash
#SBATCH --job-name=explore
#SBATCH -p cpuq
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -c 4
#SBATCH --mem-per-cpu 1800M

GIT_ROOT=$(git rev-parse --show-toplevel)
MODEL_DIR=../../linear-probes/
source "${GIT_ROOT}/activate"
# julia --project --startup-file=no -e 'using Pkg; Pkg.instantiate()'
find $MODEL_DIR -maxdepth 1 -mindepth 1 -print0 | \
    xargs -0 -P ${SLURM_NTASKS} -i ./explore_probes.jl {}

# Archive results
find $MODEL_DIR -name '*.jld2' -printf '%P\n' | \
    tar -caf linear_probes.tar.xz -C $MODEL_DIR --files-from=-
