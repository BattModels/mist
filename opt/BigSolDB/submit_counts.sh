#!/bin/bash
#SBATCH --job-name=bigsoldb-tokens
#SBATCH --partition venkvis-cpu
#SBATCH --qos=venkvis-short
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G

# Measured ~118M lines/min; the full train split is ~16 min. Runtime is set by
# gzip, which is a single sequential stream, so stride does not change it.
# Counts are written beside the archive, outside the repository.
set -ex

GIT_ROOT=$(git rev-parse --show-toplevel)
export HF_HOME="$GIT_ROOT/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH="$GIT_ROOT"
source "$GIT_ROOT/.venv/bin/activate"

ARCHIVE="${ARCHIVE:-/scratch/venkvis_root/venkvis/abhutani/realspace_v4_dev2.tar.gz}"

python "$GIT_ROOT/opt/BigSolDB/token_counts.py" corpus "$ARCHIVE" \
    --split "${SPLIT:-train}" \
    --stride "${STRIDE:-1000}" \
    ${OUTPUT:+--output "$OUTPUT"}
