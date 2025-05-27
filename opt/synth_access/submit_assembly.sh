#!/bin/bash
#SBATCH --job-name=synth-access
#SBATCH --cpus-per-task=96
#SBATCH --time=1-0:0:0
#SBATCH --partition venkvis-cpu

GIT_ROOT=$(git rev-parse --show-toplevel)
export HF_HOME="$GIT_ROOT/.cache/huggingface"
source .venv/bin/activate
apptainer exec \
    --bind /scratch,/nfs/turbo,/tmp \
    /nfs/turbo/coe-venkvis/mist/mist+pytorch+25.01+v4.sif \
/opt/uv/uv run python main.py \
    --metric 'assembly-index' \
    --output-format '{dataset}-asm.json'
