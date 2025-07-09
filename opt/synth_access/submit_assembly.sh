#!/bin/bash
#SBATCH --job-name=synth-access
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=1800M
#SBATCH --time=2-0:0:0
#SBATCH --partition venkvis-cpu,venkvis-largemem

GIT_ROOT=$(git rev-parse --show-toplevel)
module purge
module load python/3.11.5
export HF_HOME="$GIT_ROOT/.cache/huggingface"
uv run --python $(which python) python main.py \
    --metric 'assembly-index' \
    --output-format '{dataset}-asm.json' \
    --num-proc 1
