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
source ./activate
env

# # Start Tracy Profiler
# module --ignore_cache load spack
# module --ignore_cache load tracy
# export TRACY_WORKLOAD="tracy_${SLURM_JOB_ID}.tracy"
# export TRACY_PORT=$(( 9000 + $SLURM_JOB_ID % 1024 ))
# export TRACY_ENABLE=1
# tracy-capture --output-path $TRACY_WORKLOAD --port $TRACY_PORT

# Launch the job
srun --mpi=pmix \
    julia --project --threads=${SLURM_CPUS_PER_TASK:-1} --color=no --startup-file=no -- \
    ./main.jl $@

exit_code=$?
echo "`date`: done"
exit $exit_code
