#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH -A che210007p
#SBATCH --mem-per-cpu 1G
#SBATCH -c 4
#SBATCH -n 32
#SBATCH --time 4:0:0
set -x

my_job_header

# Activate the environment
module load python/3.11.5
source ./activate

MEM_PER_TASK="$(($SLURM_MEM_PER_CPU * $SLURM_CPUS_PER_TASK))M"
export PYARROW_IGNORE_TIMEZONE=1

# Include Modules from HuggingFace's Hub
export PYTHONPATH=~/.cache/huggingface/modules/
spark-submit \
    --num-executors $SLURM_NTASKS \
    --driver-memory $MEM_PER_TASK \
    --executor-memory $MEM_PER_TASK \
    --executor-cores $SLURM_CPUS_PER_TASK \
    electrolyte_fm/tokenize/tokenizer_stats.py --limit 16 --output "stats/$1/$SLURM_JOBID.json" $1 $2
