{
  nodes: 1,
  gpus_per_node: 1,
  queue: 'venkvis-debug',
  container: '/nfs/turbo/coe-venkvis/mist/mist+pytorch+25.01+v4.sif',
  train: {
    trainer: {
      devices: $.gpus_per_node,
      num_nodes: $.nodes,
    },
  },
  env: {
    JOBID: '$SLURM_JOB_ID',
  },
}
