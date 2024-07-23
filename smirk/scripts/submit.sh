#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH -n 128
#SBATCH -A che210007p
#SBATCH --mem-per-cpu 1G
#SBATCH -c 1
#SBATCH --time 4:0:0

# Activate the environment
module load python/3.11.5
source ./activate

python smirk/scripts/train.py \
    --output "./smirk-gpe-$SLURM_JOBID" \
    $@
