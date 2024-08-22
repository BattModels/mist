#!/bin/bash
#SBATCH -p "venkvis-cpu"
#SBATCH --cpus-per-task 1
#SBATCH --ntasks 48
#SBATCH --mem-per-cpu 1800M
#SBATCH --time 4:0:0
set -x
my_job_header

# Move to git root
cd "$(git rev-parse --show-toplevel)/opt/TokenizerStats"

# Activate Environment
module purge
module --ignore_cache load gcc python/3.11.5 openmpi/4.1.6
source ./activate
export TOKENIZERS_PARALLELISM=false
srun --mpi=pmix ./main.jl $@
echo "`date`: done"

