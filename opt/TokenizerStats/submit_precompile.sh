#!/bin/bash
#SBATCH -p venkvis-cpu
#SBATCH --cpus-per-task 4
#SBATCH --mem-per-cpu 1800M
#SBATCH --time=0:20:00
my_job_header

# Move to git root
cd "$(git rev-parse --show-toplevel)/opt/TokenizerStats"

# Activate Environment
source ./activate
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_PKG_USE_CLI_GIT=true
export JULIA_NUM_PRECOMPILE_TASKS=$SLURM_CPUS_ON_NODE
env

julia --color=no --startup-file=no --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
julia --color=no --startup-file=no --project -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile(timing=true)'
julia --color=no --startup-file=no --project=plots -e 'using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.precompile(timing=true)'
