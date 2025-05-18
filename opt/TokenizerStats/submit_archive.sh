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
stats_archive="$(realpath ~/scratch)/tokenizer_stats_$(date +"%d%m%Y").tar.xz"
tar -cavf  $stats_archive \
    --xz \
    --exclude="*.slurm" \
    --exclude="*.tmp" \
    --exclude='*/.unmerged/*'
    -C stats \
    .
mkdir -p ./archive
ln -sf $stats_archive ./archive/ngram_tokenizer_stats.tar.xz

# Compress Code (TokenizerStats)
code_archive="$(realpath ~/scratch)/TokenizerStats.jl_$(date +"%d%m%Y").tar.xz"
{ git ls-files; find fig/ -type f |  sed 's|^\./||'; find -name Manifest.toml | sed 's|^\./||'; } | \
    tar -cavf $code_archive \
    --dereference \
    --files-from -
ln -sf $code_archive ./archive/TokenizerStats.tar.xz

# Compress Code (Smirk)
SMIRK_VERSION="v0.1.1"
curl -L https://github.com/BattModels/smirk/archive/refs/tags/$(SMIRK_VERSION).tar.gz \
    --output "./archive/smirk_${SMIRK_VERSION}.tar.gz"
