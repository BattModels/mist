// Task Specification for CheMixHub Benchmarks
{
  drug_solubility: {
    path: '/home/abhutani/chemixhub/datasets/chemixhub_mist/drug-solubility',
    temperature: 'concat',
    target_columns: ['LogS'],
    transform: 'identity',
    metrics: ['mae'],
    n_components: 3,
  },
}
