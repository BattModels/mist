#!/bin/bash
#SBATCH -p venkvis-cpu
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time 0:30:0

set -x
my_job_header

# Move to git root
cd "$(git rev-parse --show-toplevel)/opt/TokenizerStats"

# Activate Environment
module purge
module --ignore_cache load python/3.11.5
source ./activate
env

python src/atomic_oov.py $@

exit_code=$?
echo "`date`: done"
exit $exit_code
