{
  queue: 'prod',
  nodes: 16,
  walltime: '3:00:00',
  mpiexe: {
    np: '$NGPUs',
    ppn: '$NGPU_PER_HOST',
    hostfile: '$PBS_NODEFILE',
    'cpu-bind': 'numa',
  },
  train: {
    init_args: {
      path: '/eagle/FoundEnergy/realspace_v2',
    },
  },
  env: {
    // NCCL Setting
    NCCL_DEBUG: 'WARN',
    NCCL_COLLNET_ENABLE: 1,

    // MPICH Error iff enabled: issue #30
    MPICH_GPU_SUPPORT_ENABLED: 0,

    // Get Slingshot fabric report
    MPICH_OFI_CXI_COUNTER_REPORT: 2,

    // Helper for consist JOBIDs
    JOBID: '$PBS_JOBID',
  },
}
