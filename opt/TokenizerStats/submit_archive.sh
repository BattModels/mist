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
    --exclude='*/.unmerged/*'
    -C stats \
    .

# Compress Code (TokenizerStats)
{ git ls-files; find fig/ -type f |  sed 's|^\./||'; find -name Manifest.toml | sed 's|^\./||'; } | \
    tar -cav \
    -f "$(realpath ~/scratch)/TokenizerStats.jl_$(date +"%d%m%Y").tar.xz" \
    --dereference \
    --files-from -

# Compress Code (Smirk)
SMIRK_VERSION="v0.1.1"
curl -L https://github.com/BattModels/smirk/archive/refs/tags/$(SMIRK_VERSION).tar.gz \
    --output "$(realpath ~/scratch)/smirk_${SMIRK_VERSION}.tar.gz"
