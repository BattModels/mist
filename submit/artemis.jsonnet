{
  nodes: 1,
  gpus_per_node: 1,
  queue: 'venkvis-debug',
  train: {
    data: {
      init_args: {
        path: '/nfs/turbo/coe-venkvis/mist/realspace_v4_dev/',
      },
    },
    trainer: {
      devices: $.gpus_per_node,
      num_nodes: $.nodes,
    },
  },
  env: {
    JOBID: '$SLURM_JOB_ID',
  },
}
