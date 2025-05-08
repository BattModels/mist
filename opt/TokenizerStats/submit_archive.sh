#!/bin/bash
#SBATCH -p venkvis-cpu
#SBATCH --cpus-per-task 3
#SBATCH --mem-per-cpu 1800M
#SBATCH --time=8:0:00
set -x
my_job_header

# Add tokenizers.json
cp tokenizers.json stats/

# Compress stats
export XZ_OPT="-9 -v --extreme --memlimit=4000000000 --threads=${SLURM_CPUS_PER_TASK}"
tar -cavf "$(realpath ~/scratch)/tokenizer_stats_$(date +"%d%m%Y").tar.xz" \
    --xz \
    --exclude="*.slurm" \
    --exclude="*.tmp" \
    -C stats \
    .
