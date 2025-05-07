#!/bin/bash
#SBATCH -p venkvis-cpu
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 1800M
#SBATCH --time=1:0:00
set -x
my_job_header

# Add tokenizers.json
cp tokenizers.json stats/

# Compress stats
export XZ_OPT="-e9 --memlimit=4000000000 --threads=${SLURM_CPUS_PER_TASK}"
tar -cavf "$(realpath ~/scratch)/tokenizer_stats_$(date +"%d%m%Y").tar.xz" \
    --xz \
    --exclude="*.slurm" \
    --exclude="*.tmp" \
    --exclude=".unmerged/" \
    -C stats \
    .
