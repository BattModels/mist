#!/bin/bash
#SBATCH -N 1
#SBATCH -n 48
#SBATCH -c 2
#SBATCH --mem-per-cpu 1900M
#SBATCH --time 2:0:0

my_job_header

# Activate the environment
spack env activate .
source ./activate

MEM_PER_TASK="$(($SLURM_MEM_PER_CPU * $SLURM_CPUS_PER_TASK))M"
export PYARROW_IGNORE_TIMEZONE=1
spark-submit \
    --num-executors $SLURM_NTASKS \
    --driver-memory $MEM_PER_TASK \
    --executor-memory $MEM_PER_TASK \
    --executor-cores $SLURM_CPUS_PER_TASK \
    electrolye_fm/tokenize/tokenizer_stats.py --output "$1/stats-$SLURM_JOBID.json" $1 $2
