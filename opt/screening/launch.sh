#!/bin/bash
#SBATCH --time 0:30:0
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-node 8
#SBATCH --cpus-per-task 12
#SBATCH --mem-per-cpu 1800M
#SBATCH --export=NONE

export PMIX_MCA_gds=hash
export NCCL_TOPO_FILE=/cm/shared/etc/ndv4-topo.xml
export MELLANOC_VISIBLE_DEVICES=all
env
set -ex

srun --mpi=pmix \
$(which apptainer) run \
--nv \
--bind /lustre/fs0,$TMPDIR \
--env PYTHONPATH="../../" \
/lustre/fs0/shared/sqsh-files/mist+pytorch+25.01+v4.sif \
../../submit/set_node_rank \
python screen.py --gpus-per-node ${SLURM_GPUS_PER_NODE:-1} --num-nodes ${SLURM_NNODES:-1} $@
# nsys profile \
#     --output="nsys_multinode_%q{JOBID}_%q{NODE_RANK}" \
#     --trace=cuda,cudnn,cublas,nvtx \
#     --cuda-memory-usage=true \
#     --duration 120 \
