#!/bin/bash
#SBATCH --job-name=omol25
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=1800M
#SBATCH --partition=venkvis-debug
my_job_header
set -ex

./omol25.py \
    --output $(realpath ~/scratch/omol25-ds) \
    --num-proc ${SLURM_CPUS_PER_TASK} \
    --read-workers 6  \
    $(realpath ../../omol25)

echo "done: $(date)"
