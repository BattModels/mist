#!/bin/bash
#SBATCH -p venkvis-cpu
#SBATCH --cpus-per-task 1
#SBATCH --ntasks 16
#SBATCH --mem-per-cpu 1800M
set -x
my_job_header

# Move to git root
cd "$(git rev-parse --show-toplevel)/opt/TokenizerStats"

# Activate Environment
module purge
module --ignore_cache load gcc python/3.11.5 openmpi/4.1.6
source ./activate
export TOKENIZERS_PARALLELISM=false
env

# module load cuda/12.2.1
# nsys profile -o "nsys_multinode_%q{SLURM_JOB_ID}_%q{PMIX_RANK}" --trace=mpi,nvtx \
srun --mpi=pmix \
    julia --project --threads=${SLURM_CPUS_PER_TASK:-1} --color=no --startup-file=no -- \
    ./main.jl $@

exit_code=$?
echo "`date`: done"
exit $exit_code
