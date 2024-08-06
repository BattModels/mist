#!/bin/bash
#SBATCH -p venkvis-cpu
#SBATCH -c 64
#SBATCH --mem-per-cpu=1800M
#SBATCH --time 12:0:0

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-auto}
julia --project=opt/BayesianScaling opt/BayesianScaling/main.jl $@
