#!/bin/bash
#SBATCH -p cpuq
#SBATCH -c 32
#SBATCH --mem 126G
#SBATCH --time 6:0:0
set -x
my_job_header

# Move to git root
cd "$(git rev-parse --show-toplevel)/opt/TokenizerStats"

# Activate Environment
module purge
module --ignore_cache load gcc python/3.11.5 openmpi/4.1.6
source ./activate
export TOKENIZERS_PARALLELISM=false
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'

# Copy files to /tmp
mkdir -p /tmp/SmilesPE
find stats/SmilesPE/SPE_ChEMBL -name 'realspace_v4_dev2.bson*' -print0 | xargs -0 -P 32 -i cp -v {} /tmp/SmilesPE/

# Merge files
julia --project --color=no --startup-file=no -- merge.jl

# Cleanup
rm -rf /tmp/SmilesPE
