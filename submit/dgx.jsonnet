{
  stage: 'fit',
  nodes: 2,
  gpus_per_node: 8,
  container: '/lustre/fs0/shared/sqsh-files/mist+pytorch+25.01+v4.sif',
  train: {
    trainer: {
      devices: $.gpus_per_node,
      num_nodes: $.nodes,
    },
  },
  env: {
    JOBID: '$SLURM_JOB_ID',
    PMIX_MCA_gds: 'hash',
    NCCL_TOPO_FILE: '/cm/shared/etc/ndv4-topo.xml',
    MELLANOC_VISIBLE_DEVICES: 'all',
    MASTER_PORT: '$(( 53394 + $SLURM_JOB_ID % 1024 ))',
  },
}
