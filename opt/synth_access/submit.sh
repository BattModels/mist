#!/bin/bash
#SBATCH --job-name=synth-access
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=1:00:0
#SBATCH --partition venkvis-h100,venkvis-a100

GIT_ROOT=$(git rev-parse --show-toplevel)
export HF_HOME="$GIT_ROOT/.cache/huggingface"
source .venv/bin/activate
apptainer exec \
    --bind /scratch,/nfs/turbo,/tmp \
    /nfs/turbo/coe-venkvis/mist/mist+pytorch+25.01+v4.sif \
/opt/uv/uv run python main.py
