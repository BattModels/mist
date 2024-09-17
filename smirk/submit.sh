#!/bin/bash
#SBATCH -c 64
#SBATCH --time 2-0:0:0
#SBATCH --mem-per-cpu 3G
#SBATCH -p venkvis-largemem

# Move to git root
cd $(git rev-parse --show-toplevel)

# Activate Environment
module purge
module --ignore_cache load spack/0.21 python/3.11.5 cuda/12.2 openmpi/4.1.6
source ./activate

set -x
python -m smirk.cli \
    --vocab-size 50000 \
    --merge-brackets \
    --split-structure \
    --output smirk-gpe-50k-mb-ss \
    /home/awadell/turbo/../mist/realspace_v4_dev/data/train/part-00000-7ad465b7-d510-40d8-b5db-9012116315d1-c000.txt \
    /home/awadell/turbo/../mist/realspace_v4_dev/data/train/part-00001-7ad465b7-d510-40d8-b5db-9012116315d1-c000.txt \
    /home/awadell/turbo/../mist/realspace_v4_dev/data/train/part-00002-7ad465b7-d510-40d8-b5db-9012116315d1-c000.txt \
    /home/awadell/turbo/../mist/realspace_v4_dev/data/train/part-00003-7ad465b7-d510-40d8-b5db-9012116315d1-c000.txt \
    /home/awadell/turbo/../mist/realspace_v4_dev/data/train/part-00004-7ad465b7-d510-40d8-b5db-9012116315d1-c000.txt \
    /home/awadell/turbo/../mist/realspace_v4_dev/data/train/part-00005-7ad465b7-d510-40d8-b5db-9012116315d1-c000.txt \
    /home/awadell/turbo/../mist/realspace_v4_dev/data/train/part-00006-7ad465b7-d510-40d8-b5db-9012116315d1-c000.txt \
    /home/awadell/turbo/../mist/realspace_v4_dev/data/train/part-00007-7ad465b7-d510-40d8-b5db-9012116315d1-c000.txt \
    /home/awadell/turbo/../mist/realspace_v4_dev/data/train/part-00008-7ad465b7-d510-40d8-b5db-9012116315d1-c000.txt \
    /home/awadell/turbo/../mist/realspace_v4_dev/data/train/part-00009-7ad465b7-d510-40d8-b5db-9012116315d1-c000.txt
