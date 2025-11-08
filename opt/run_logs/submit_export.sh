#!/bin/bash
#SBATCH --account bcuf-delta-gpu
#SBATCH --job-name=wandb_export
#SBATCH --output=logs/wandb_export_%A_%a.out
#SBATCH --error=logs/wandb_export_%A_%a.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --constraint="scratch"
#SBATCH --mem=32G
#SBATCH --array=0-25

# Export script for delta
set -ex

# Move to git root
cd $(git rev-parse --show-toplevel)

# Activate Environment
module purge
module --ignore_cache load python/3.11.6 openmpi/4.1.6 cuda/12.2.1
source ./activate

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export WANDB_API_KEY=${WANDB_API_KEY}  # Make sure this is set in your environment
export WANDB_CACHE_DIR=/tmp/wandb_cache_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}

# Create temp cache directory
mkdir -p $WANDB_CACHE_DIR

# Run the export script
python opt/run_logs/sync_wandb.py \
    --slurm-array-id $SLURM_ARRAY_TASK_ID \
    --slurm-array-size $SLURM_ARRAY_TASK_COUNT \
    --n-jobs 3 \
    --entity incite-mist \
    --project mist

rm -rf $WANDB_CACHE_DIR
echo "Array task $SLURM_ARRAY_TASK_ID completed"