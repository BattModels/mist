{
  nodes: 1,
  gpus_per_node: 1,
  queue: 'venkvis-debug',
  train: {
    trainer: {
      devices: $.gpus_per_node,
      num_nodes: $.nodes,
    },
  },
  env: {
    JOBID: '$SLURM_JOB_ID',
    MASTER_PORT: '$(( 53394 + $SLURM_JOB_ID % 1024 ))',
  },
}
