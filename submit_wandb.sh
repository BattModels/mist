#!/bin/bash
#SBATCH --account bcuf-delta-gpu
#SBATCH -p gpuA100x4
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-node 1
#SBATCH --time 48:00:00
#SBATCH --constraint="scratch"
#SBATCH --cpus-per-task 16
#SBATCH --mem-per-cpu 4G
#SBATCH --array=1-50
#SBATCH --signal=USR1@90
#SBATCH --requeue
#SBATCH --open-mode=append

my_job_header
set -ex

# Activate Environment
module purge
module --ignore_cache load python/3.11.6 openmpi/4.1.6 cuda/12.2.1
source ./activate

export TMPDIR=${TMPDIR:-/tmp}

# Set env variables
ENV_FILE="${TMPDIR}/env-${SLURM_JOB_ID}.sh"
cat > $ENV_FILE<<EOF
EOF
source $ENV_FILE
sbcast -fp $ENV_FILE $ENV_FILE

wandb agent --count 1 incite-mist/electrolyte-fm/fwsj26wi