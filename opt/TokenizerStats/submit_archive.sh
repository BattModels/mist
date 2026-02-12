#!/bin/bash
#SBATCH -p venkvis-cpu
#SBATCH --cpus-per-task 8
#SBATCH --mem-per-cpu 2G
#SBATCH --time=8:0:00
set -ex
my_job_header

# Locate directory with tokenizers stats
STATS_DIR=$(realpath stats/)

# Add tokenizers.json
cp tokenizers.json $STATS_DIR

# # Compress stats
stats_archive="$(realpath ~/scratch)/tokenizer_stats_$(date +"%d%m%Y").tar.gz"
tar -cavf  "$stats_archive" \
    --exclude="*.slurm" \
    --exclude="*.tmp" \
    --exclude='*/.unmerged/*' \
    -C "$STATS_DIR" \
    .
mkdir -p ./archive
ln -sf $stats_archive ./archive/ngram_tokenizer_stats.tar.gz

# Compress Code (TokenizerStats)
code_archive="$(realpath ~/scratch)/TokenizerStats.jl_$(date +"%d%m%Y").tar.gz"
{ git ls-files; find fig/ -type f |  sed 's|^\./||'; find -name Manifest.toml | sed 's|^\./||'; } | \
    tar -cavf $code_archive \
    --dereference \
    --files-from -
ln -sf $code_archive ./archive/TokenizerStats.tar.gz

# Compress Code (Smirk)
SMIRK_VERSION="v0.1.1"
curl -L https://github.com/BattModels/smirk/archive/refs/tags/$(SMIRK_VERSION).tar.gz \
    --output "./archive/smirk_${SMIRK_VERSION}.tar.gz"

# Archive to DataDen
SRC="3242c149-a2b9-4dba-9406-ae3717981621" # Artemis
DST="ab65757f-00f5-4e5b-aa21-133187732a01" # DataDen
if [[ $(type -t globus) ]]; then
    find $(realpath ./archive) -not -type d -printf '%f %f\n' | \
        globus transfer --batch - \
        $SRC:$(realpath ./archive) \
        $DST:"/coe-venkvis/awadell/smirk-paper-archive_$(date +"%d%m%Y")/"
fi
