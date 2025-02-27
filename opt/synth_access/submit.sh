#!/bin/bash
#SBATCH --job-name=synth-access
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:0
#SBATCH --partition venkvis-h100,venkvis-a100

GIT_ROOT=$(git rev-parse --show-toplevel)
export HF_HOME="$GIT_ROOT/.cache/huggingface"
source .venv/bin/activate

python main.py
