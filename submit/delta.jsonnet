{
  nodes: 1,
  gpus_per_node: 1,
  queue: 'gpuA100x4',
  train: {
    data: {
      init_args: {
        path: '/projects/bcuf/',
      },
    },
    trainer: {
      devices: $.gpus_per_node,
      num_nodes: $.nodes,
    },
  },
  env: {
    JOBID: '$SLURM_JOB_ID',
    LD_LIBRARY_PATH: '$LD_LIBRARY_PATH:$LIBRARY_PATH',
  },
}
