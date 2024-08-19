#!/bin/bash
#SBATCH -p "venkvis-cpu"
#SBATCH --cpus-per-task 2
#SBATCH --ntasks 48
#SBATCH --mem-per-cpu 1800M
#SBATCH --time 4:0:0

# Move to git root
cd $(git rev-parse --show-toplevel)

# Activate Environment
module purge
module --ignore_cache load spack/0.21 python/3.11.5 openmpi/4.1.6
source ./activate

srun --mpi=pmix \
    julia --project=opt/TokenizerStats --startup-file=no -- opt/TokenizerStats/tokenizer_stats.jl $@

